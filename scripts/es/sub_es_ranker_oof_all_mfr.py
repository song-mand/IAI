#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/es/sub_es_ranker_oof_all_mfr.py

Submission code.

Strategy:
  - ES: RF detector + OOF candidate ranker + margin decision + optional ByT5 candidate reranking
  - Other languages: MFR

Required files:
  - scripts/es/es_rules.py
  - ./final_model/es_model
  - ./detectors/es_resources_oof.joblib
  - ./detectors/es_change_detector_rf.joblib
  - ./detectors/es_candidate_ranker_oof_rf.joblib

Important change:
  - Normal candidates use --candidate_margin.
  - ByT5 candidates use separate --byt5_candidate_margin.
  - This lets filtered ByT5 outputs pass more easily when the OOF ranker is too conservative.
"""

import argparse
import gc
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

import joblib
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from es_rules import (  # noqa: E402
    candidate_features,
    detector_features,
    generate_candidates,
    is_protected_token,
    safe_byt5_output,
    target_of,
)

LANG = "es"

ALL_LANGS = [
    "en", "da", "de", "es", "hr", "it", "nl", "sl", "sr", "tr", "iden", "trde",
    "id", "ja", "ko", "th", "vi",
]


def build_all_mfr(train_split):
    counts_by_lang = defaultdict(lambda: defaultdict(Counter))

    for row in train_split:
        lang = row["lang"]

        for raw, norm in zip(row["raw"], row["norm"]):
            counts_by_lang[lang][raw][target_of(raw, norm)] += 1

    mfr_by_lang = {}

    for lang, counts in counts_by_lang.items():
        mfr = {}

        for raw, counter in counts.items():
            best, _best_count = max(
                counter.items(),
                key=lambda x: (x[1], x[0] == raw),
            )
            mfr[raw] = best

        mfr_by_lang[lang] = mfr

    return mfr_by_lang


def predict_mfr(raw_words, mfr_dict):
    return [mfr_dict.get(w, w) for w in raw_words]


def generate_candidates_wrapper(raw, resources):
    return generate_candidates(
        raw,
        mfr=resources["mfr"],
        mfr_conf=resources["mfr_conf"],
        key_map=resources["key_map"],
    )


def score_candidates(raw, left, right, cands, ranker, resources):
    feats = [
        candidate_features(raw, cand, source, left, right, resources)
        for source, cand in cands
    ]

    probs = ranker.predict_proba(feats)[:, 1]

    copy_score = None
    best_noncopy = None

    for (source, cand), score in zip(cands, probs):
        score = float(score)

        if cand == raw:
            if copy_score is None or score > copy_score:
                copy_score = score
        else:
            if best_noncopy is None or score > best_noncopy["score"]:
                best_noncopy = {
                    "source": source,
                    "cand": cand,
                    "score": score,
                }

    if copy_score is None:
        copy_score = 0.0

    return copy_score, best_noncopy


def build_byt5_input(raw_words, token_idx, prompt_format):
    raw = raw_words[token_idx]

    if prompt_format == "natural":
        return f"lang: es word: {raw} context: {' '.join(raw_words)}"

    if prompt_format == "marked_natural":
        words_copy = list(raw_words)
        words_copy[token_idx] = f"<extra_id_0> {raw} <extra_id_1>"
        return f"normalize lang: es target: {raw} context: {' '.join(words_copy)}"

    words_copy = list(raw_words)
    words_copy[token_idx] = f"<extra_id_0> {raw} <extra_id_1>"
    return " ".join(words_copy)


def rerank_with_byt5_margin(
    raw,
    left,
    right,
    base_cands,
    byt5_pred,
    ranker,
    resources,
    candidate_margin,
    byt5_candidate_margin,
):
    """
    Add ByT5 output as source='byt5', then rerank.

    Normal candidates:
      accept if margin >= candidate_margin

    ByT5 candidate:
      accept if margin >= byt5_candidate_margin

    This is intentionally separate because OOF ranker is conservative and often
    under-scores ByT5 candidates.
    """
    cands = list(base_cands)

    if byt5_pred not in [cand for _source, cand in cands]:
        cands.append(("byt5", byt5_pred))

    feats = [
        candidate_features(raw, cand, source, left, right, resources)
        for source, cand in cands
    ]

    probs = ranker.predict_proba(feats)[:, 1]

    copy_score = None
    best_noncopy = None
    best_byt5 = None

    for (source, cand), score in zip(cands, probs):
        score = float(score)

        if cand == raw:
            if copy_score is None or score > copy_score:
                copy_score = score
            continue

        item = {
            "source": source,
            "cand": cand,
            "score": score,
        }

        if best_noncopy is None or score > best_noncopy["score"]:
            best_noncopy = item

        if source == "byt5":
            if best_byt5 is None or score > best_byt5["score"]:
                best_byt5 = item

    if copy_score is None:
        copy_score = 0.0

    accepted = None

    # 1. If the best non-copy is not ByT5, use normal candidate margin.
    if best_noncopy is not None and best_noncopy["source"] != "byt5":
        normal_margin = best_noncopy["score"] - copy_score

        if normal_margin >= candidate_margin:
            accepted = best_noncopy

    # 2. ByT5 uses separate, usually lower margin.
    if accepted is None and best_byt5 is not None:
        byt5_margin = best_byt5["score"] - copy_score

        if byt5_margin >= byt5_candidate_margin:
            accepted = best_byt5

    # 3. If the overall best non-copy is ByT5, also apply ByT5 margin.
    if accepted is None and best_noncopy is not None and best_noncopy["source"] == "byt5":
        byt5_margin = best_noncopy["score"] - copy_score

        if byt5_margin >= byt5_candidate_margin:
            accepted = best_noncopy

    return accepted, copy_score, best_noncopy, best_byt5


def predict_es_rows_batched(
    es_rows,
    resources,
    detector,
    ranker,
    detector_threshold,
    candidate_margin,
    use_byt5,
    byt5_copy_detector_threshold,
    byt5_candidate_margin,
    tokenizer,
    model,
    device,
    prompt_format,
    max_length,
    max_new_tokens,
    num_beams,
    byt5_generate_batch_size,
):
    predictions_by_row = []
    requests = []
    counts = Counter()

    for row_idx, row in enumerate(tqdm(es_rows, desc="[ES] detector/ranker pass")):
        raw_words = list(row["raw"])
        pred_words = list(raw_words)

        for i, raw in enumerate(raw_words):
            if is_protected_token(raw):
                counts["protected_copy"] += 1
                continue

            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

            dprob = detector.predict_proba(
                [detector_features(raw, left, right, resources)]
            )[0][1]

            if dprob < detector_threshold:
                counts["detector_copy"] += 1
                continue

            cands = generate_candidates_wrapper(raw, resources)

            copy_score, best_noncopy = score_candidates(
                raw=raw,
                left=left,
                right=right,
                cands=cands,
                ranker=ranker,
                resources=resources,
            )

            margin = (
                best_noncopy["score"] - copy_score
                if best_noncopy is not None
                else -1.0
            )

            if best_noncopy is not None and margin >= candidate_margin:
                pred_words[i] = best_noncopy["cand"]
                counts["ranker_margin_change"] += 1
                counts[f"ranker_source_{best_noncopy['source']}"] += 1
                continue

            if use_byt5 and dprob >= byt5_copy_detector_threshold:
                requests.append({
                    "row_idx": row_idx,
                    "token_idx": i,
                    "raw": raw,
                    "left": left,
                    "right": right,
                    "base_cands": cands,
                    "detector_prob": float(dprob),
                    "copy_score": float(copy_score),
                    "best_noncopy": best_noncopy,
                })
                counts["byt5_requested"] += 1
            else:
                counts["ranker_margin_copy"] += 1

        predictions_by_row.append(pred_words)

    print("\n[ES] decision counts before ByT5:", flush=True)
    for k, v in counts.most_common():
        print(f"  {k:32s} {v}", flush=True)
    print(f"[ES] total ByT5 requested tokens: {len(requests)}", flush=True)

    if not use_byt5 or not requests:
        return predictions_by_row, counts

    all_inputs = []

    for req in requests:
        raw_words = list(es_rows[req["row_idx"]]["raw"])
        all_inputs.append(
            build_byt5_input(raw_words, req["token_idx"], prompt_format)
        )

    for start in tqdm(
        range(0, len(all_inputs), byt5_generate_batch_size),
        desc="[ES] batched ByT5",
    ):
        end = min(start + byt5_generate_batch_size, len(all_inputs))
        batch_inputs = all_inputs[start:end]
        batch_reqs = requests[start:end]

        batch = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for req, decoded_pred in zip(batch_reqs, decoded):
            row_idx = req["row_idx"]
            token_idx = req["token_idx"]
            raw = req["raw"]
            left = req["left"]
            right = req["right"]

            pred = safe_byt5_output(raw, decoded_pred, allow_underscore=True)

            if pred is None:
                counts["byt5_invalid_copy"] += 1
                continue

            accepted, copy_score, best_noncopy, best_byt5 = rerank_with_byt5_margin(
                raw=raw,
                left=left,
                right=right,
                base_cands=req["base_cands"],
                byt5_pred=pred,
                ranker=ranker,
                resources=resources,
                candidate_margin=candidate_margin,
                byt5_candidate_margin=byt5_candidate_margin,
            )

            if accepted is not None:
                predictions_by_row[row_idx][token_idx] = accepted["cand"]
                counts["byt5_rerank_change"] += 1
                counts[f"byt5_rerank_source_{accepted['source']}"] += 1
            else:
                counts["byt5_rerank_copy"] += 1

    print("\n[ES] final decision counts:", flush=True)
    for k, v in counts.most_common():
        print(f"  {k:32s} {v}", flush=True)

    return predictions_by_row, counts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")

    parser.add_argument("--es_model_dir", type=str, default="./final_model/es_model")
    parser.add_argument("--es_detector", type=str, default="./detectors/es_change_detector_rf.joblib")
    parser.add_argument("--es_ranker", type=str, default="./detectors/es_candidate_ranker_oof_rf.joblib")
    parser.add_argument("--es_resources", type=str, default="./detectors/es_resources_oof.joblib")

    parser.add_argument("--output_json", type=str, default="./submission_files/predictions.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip")

    parser.add_argument("--detector_threshold", type=float, default=-1.0)
    parser.add_argument("--candidate_margin", type=float, default=-1.0)

    parser.add_argument("--use_byt5", action="store_true")
    parser.add_argument("--disable_byt5", dest="use_byt5", action="store_false")
    parser.set_defaults(use_byt5=True)

    parser.add_argument("--byt5_copy_detector_threshold", type=float, default=0.50)

    parser.add_argument(
        "--byt5_candidate_margin",
        type=float,
        default=-0.05,
        help="Separate margin for accepting filtered ByT5 candidates.",
    )

    parser.add_argument(
        "--prompt_format",
        choices=["sentinel", "natural", "marked_natural"],
        default="sentinel",
    )

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--byt5_generate_batch_size", type=int, default=64)

    args = parser.parse_args()

    print("Loading dataset...", flush=True)
    dataset = load_dataset(args.dataset_name)
    train_split = dataset["train"]
    test_split = dataset["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Building MFR dictionaries for non-ES languages...", flush=True)
    mfr_by_lang = build_all_mfr(train_split)

    for lang in ALL_LANGS:
        print(f"[{lang.upper()}] MFR entries: {len(mfr_by_lang.get(lang, {}))}", flush=True)

    use_es_hybrid = (
        os.path.isfile(args.es_detector)
        and os.path.isfile(args.es_ranker)
        and os.path.isfile(args.es_resources)
    )

    if not use_es_hybrid:
        print("[ES] hybrid files missing. ES will use MFR only.", flush=True)

    if use_es_hybrid:
        resources = joblib.load(args.es_resources)
        det_art = joblib.load(args.es_detector)
        rank_art = joblib.load(args.es_ranker)

        detector = det_art["model"]
        ranker = rank_art["model"]

        detector_threshold = (
            args.detector_threshold
            if args.detector_threshold >= 0
            else det_art.get("threshold", 0.43)
        )

        candidate_margin = (
            args.candidate_margin
            if args.candidate_margin >= 0
            else rank_art.get("candidate_margin", 0.05)
        )

        print(f"[ES] detector threshold: {detector_threshold}", flush=True)
        print(f"[ES] candidate margin: {candidate_margin}", flush=True)
        print(f"[ES] byt5_copy_detector_threshold: {args.byt5_copy_detector_threshold}", flush=True)
        print(f"[ES] byt5_candidate_margin: {args.byt5_candidate_margin}", flush=True)

        tokenizer = None
        model = None

        if args.use_byt5:
            if os.path.isdir(args.es_model_dir):
                tokenizer = AutoTokenizer.from_pretrained(args.es_model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(args.es_model_dir).to(device)
                model.eval()
                print(f"[ES] ByT5 loaded: {args.es_model_dir}", flush=True)
            else:
                print("[ES] ByT5 requested but model dir missing. Disable ByT5.", flush=True)
                args.use_byt5 = False

    else:
        resources = None
        detector = None
        ranker = None
        tokenizer = None
        model = None
        detector_threshold = None
        candidate_margin = None

    es_rows = []
    es_positions = []
    normal_rows = []

    for idx, row in enumerate(test_split):
        if row["lang"] == LANG:
            es_positions.append(idx)
            es_rows.append(row)
        else:
            normal_rows.append((idx, row))

    print(f"ES rows: {len(es_rows)}", flush=True)
    print(f"Non-ES rows: {len(normal_rows)}", flush=True)

    predictions_by_idx = {}

    print("Predicting non-ES with MFR...", flush=True)

    for idx, row in tqdm(normal_rows, desc="Non-ES MFR"):
        lang = row["lang"]
        raw_words = list(row["raw"])
        pred_words = predict_mfr(raw_words, mfr_by_lang.get(lang, {}))
        predictions_by_idx[idx] = {
            "raw": raw_words,
            "pred": pred_words,
            "lang": lang,
        }

    if es_rows:
        if use_es_hybrid:
            print("Predicting ES with OOF-ranker hybrid...", flush=True)

            es_pred_words_list, _counts = predict_es_rows_batched(
                es_rows=es_rows,
                resources=resources,
                detector=detector,
                ranker=ranker,
                detector_threshold=detector_threshold,
                candidate_margin=candidate_margin,
                use_byt5=args.use_byt5,
                byt5_copy_detector_threshold=args.byt5_copy_detector_threshold,
                byt5_candidate_margin=args.byt5_candidate_margin,
                tokenizer=tokenizer,
                model=model,
                device=device,
                prompt_format=args.prompt_format,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                byt5_generate_batch_size=args.byt5_generate_batch_size,
            )

            for idx, row, pred_words in zip(es_positions, es_rows, es_pred_words_list):
                predictions_by_idx[idx] = {
                    "raw": list(row["raw"]),
                    "pred": pred_words,
                    "lang": LANG,
                }

        else:
            print("Predicting ES with MFR...", flush=True)

            for idx, row in tqdm(
                zip(es_positions, es_rows),
                total=len(es_rows),
                desc="ES MFR",
            ):
                raw_words = list(row["raw"])
                pred_words = predict_mfr(raw_words, mfr_by_lang.get(LANG, {}))
                predictions_by_idx[idx] = {
                    "raw": raw_words,
                    "pred": pred_words,
                    "lang": LANG,
                }

    print("Restoring original order...", flush=True)
    predictions = [predictions_by_idx[i] for i in range(len(test_split))]

    print("Writing submission...", flush=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.output_json, arcname="predictions.json")

    if model is not None:
        del model, tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    print("\nSaved:")
    print(f"  {args.output_json}")
    print(f"  {args.output_zip}")


if __name__ == "__main__":
    main()