#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/es/sub_es_hybrid_all_mfr.py

Full submission:
- Spanish: RF detector + candidate ranker + optional ByT5 candidate
- Other languages: MFR only
"""

import argparse
import gc
import json
import os
import zipfile
from collections import Counter, defaultdict

import joblib
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from es_rules import (
    candidate_features,
    detector_features,
    generate_candidates,
    is_protected_token,
    safe_byt5_output,
    target_of,
)

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
    conf_by_lang = {}
    for lang, counts in counts_by_lang.items():
        mfr = {}
        conf = {}
        for raw, counter in counts.items():
            total = sum(counter.values())
            best, best_count = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
            mfr[raw] = best
            conf[raw] = best_count / total if total else 0.0
        mfr_by_lang[lang] = mfr
        conf_by_lang[lang] = conf
    return mfr_by_lang, conf_by_lang


def predict_mfr(raw_words, mfr_dict):
    return [mfr_dict.get(w, w) for w in raw_words]


def build_es_byt5_input(raw_words, token_idx, prompt_format):
    raw = raw_words[token_idx]
    if prompt_format == "natural":
        context = " ".join(raw_words)
        return f"lang: es word: {raw} context: {context}"
    if prompt_format == "marked_natural":
        words_copy = list(raw_words)
        words_copy[token_idx] = f"<extra_id_0> {raw} <extra_id_1>"
        marked_context = " ".join(words_copy)
        return f"normalize lang: es target: {raw} context: {marked_context}"
    words_copy = list(raw_words)
    words_copy[token_idx] = f"<extra_id_0> {raw} <extra_id_1>"
    return " ".join(words_copy)


def predict_es_rows_batched(
    es_rows,
    resources,
    detector,
    ranker,
    detector_threshold,
    ranker_threshold,
    mfr_min_conf,
    use_byt5,
    tokenizer,
    model,
    device,
    prompt_format,
    max_length,
    max_new_tokens,
    num_beams,
    byt5_generate_batch_size,
    low_ranker_byt5_threshold,
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

            mfr = resources["mfr"]
            mfr_conf = resources["mfr_conf"]
            key_map = resources["key_map"]

            mp = mfr.get(raw, raw)
            mc = mfr_conf.get(raw, 0.0)
            if mp != raw and mc >= mfr_min_conf:
                counts["mfr_available_high_conf"] += 1
            """
            if mp != raw and mc >= mfr_min_conf:
                pred_words[i] = mp
                counts["mfr_high_conf_change"] += 1
                continue
            """

            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

            dprob = detector.predict_proba([detector_features(raw, left, right, resources)])[0][1]
            if dprob < detector_threshold:
                counts["detector_copy"] += 1
                continue

            cands = generate_candidates(raw, mfr=mfr, mfr_conf=mfr_conf, key_map=key_map)
            feats = [candidate_features(raw, cand, source, left, right, resources) for source, cand in cands]
            probs = ranker.predict_proba(feats)[:, 1]
            best_i = int(np.argmax(probs))
            best_source, best_cand = cands[best_i]
            best_score = float(probs[best_i])

            if best_cand != raw and best_score >= ranker_threshold:
                pred_words[i] = best_cand
                counts["ranker_change"] += 1
                counts[f"ranker_source_{best_source}"] += 1
                """
                elif use_byt5 and best_score < low_ranker_byt5_threshold:
                    requests.append({"row_idx": row_idx, "token_idx": i, "raw": raw})
                    counts["byt5_requested"] += 1
                """
            elif use_byt5 and (best_cand == raw or best_score < low_ranker_byt5_threshold):
                requests.append({
                    "row_idx": row_idx,
                    "token_idx": i,
                    "raw": raw,
                    "left": left,
                    "right": right,
                    "base_candidates": cands,
                })
                counts["byt5_requested"] += 1
            else:
                counts["ranker_copy"] += 1

        predictions_by_row.append(pred_words)

    print("\n[ES] decision counts before ByT5:")
    for k, v in counts.most_common():
        print(f"  {k:28s} {v}")
    print(f"[ES] total ByT5 requested tokens: {len(requests)}")

    if not use_byt5 or len(requests) == 0:
        return predictions_by_row, counts

    all_inputs = []
    for req in requests:
        row = es_rows[req["row_idx"]]
        raw_words = list(row["raw"])
        all_inputs.append(build_es_byt5_input(raw_words, req["token_idx"], prompt_format))

    for start in tqdm(range(0, len(all_inputs), byt5_generate_batch_size), desc="[ES] batched ByT5"):
        end = min(start + byt5_generate_batch_size, len(all_inputs))
        batch_inputs = all_inputs[start:end]
        batch_reqs = requests[start:end]

        batch = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

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

        for req, dec in zip(batch_reqs, decoded):
            row_idx = req["row_idx"]
            token_idx = req["token_idx"]
            raw = req["raw"]
            pred = safe_byt5_output(raw, dec, allow_underscore=True)

            if pred is None:
                predictions_by_row[row_idx][token_idx] = raw
                counts["byt5_invalid_copy"] += 1
            elif pred != raw:
                predictions_by_row[row_idx][token_idx] = pred
                counts["byt5_change"] += 1
            else:
                predictions_by_row[row_idx][token_idx] = raw
                counts["byt5_copy"] += 1

    print("\n[ES] final decision counts:")
    for k, v in counts.most_common():
        print(f"  {k:28s} {v}")

    return predictions_by_row, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--es_model_dir", type=str, default="./final_model/es_model")
    parser.add_argument("--es_detector", type=str, default="./detectors/es_change_detector_rf.joblib")
    parser.add_argument("--es_ranker", type=str, default="./detectors/es_candidate_ranker_rf.joblib")
    parser.add_argument("--es_resources", type=str, default="./detectors/es_resources.joblib")
    parser.add_argument("--output_json", type=str, default="./submission_files/predictions.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip")
    parser.add_argument("--detector_threshold", type=float, default=0.43)
    parser.add_argument("--ranker_threshold", type=float, default=0.25)
    parser.add_argument("--mfr_min_conf", type=float, default=0.75)
    parser.add_argument("--use_byt5", action="store_true")
    parser.add_argument("--disable_byt5", dest="use_byt5", action="store_false")
    parser.set_defaults(use_byt5=True)
    parser.add_argument("--low_ranker_byt5_threshold", type=float, default=0.90)
    parser.add_argument("--prompt_format", choices=["sentinel", "natural", "marked_natural"], default="sentinel")
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

    print("Building MFR dictionaries for all languages...", flush=True)
    mfr_by_lang, _conf_by_lang = build_all_mfr(train_split)
    for lang in ALL_LANGS:
        print(f"[{lang.upper()}] MFR entries: {len(mfr_by_lang.get(lang, {}))}", flush=True)

    use_es_hybrid = os.path.isfile(args.es_detector) and os.path.isfile(args.es_ranker) and os.path.isfile(args.es_resources)
    if not use_es_hybrid:
        print("[ES] hybrid files missing. ES will use MFR only.", flush=True)

    if use_es_hybrid:
        resources = joblib.load(args.es_resources)
        det_art = joblib.load(args.es_detector)
        rank_art = joblib.load(args.es_ranker)
        detector = det_art["model"]
        ranker = rank_art["model"]
        detector_threshold = args.detector_threshold if args.detector_threshold >= 0 else det_art.get("threshold", 0.48)
        ranker_threshold = args.ranker_threshold if args.ranker_threshold >= 0 else rank_art.get("ranker_threshold", 0.45)
        print(f"[ES] detector threshold: {detector_threshold}", flush=True)
        print(f"[ES] ranker threshold: {ranker_threshold}", flush=True)
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
        resources = detector = ranker = tokenizer = model = None
        detector_threshold = ranker_threshold = None

    print("Preparing test rows...", flush=True)
    es_rows, es_positions, normal_rows = [], [], []
    for idx, row in enumerate(test_split):
        if row["lang"] == "es":
            es_positions.append(idx)
            es_rows.append(row)
        else:
            normal_rows.append((idx, row))
    print(f"ES rows: {len(es_rows)}")
    print(f"Non-ES rows: {len(normal_rows)}")

    predictions_by_idx = {}
    print("Predicting non-ES with MFR...", flush=True)
    for idx, row in tqdm(normal_rows, desc="Non-ES MFR"):
        lang = row["lang"]
        raw_words = list(row["raw"])
        pred_words = predict_mfr(raw_words, mfr_by_lang.get(lang, {}))
        predictions_by_idx[idx] = {"raw": raw_words, "pred": pred_words, "lang": lang}

    if len(es_rows) > 0:
        if use_es_hybrid:
            print("Predicting ES with hybrid...", flush=True)
            es_pred_words_list, _counts = predict_es_rows_batched(
                es_rows=es_rows,
                resources=resources,
                detector=detector,
                ranker=ranker,
                detector_threshold=detector_threshold,
                ranker_threshold=ranker_threshold,
                mfr_min_conf=args.mfr_min_conf,
                use_byt5=args.use_byt5,
                tokenizer=tokenizer,
                model=model,
                device=device,
                prompt_format=args.prompt_format,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                byt5_generate_batch_size=args.byt5_generate_batch_size,
                low_ranker_byt5_threshold=args.low_ranker_byt5_threshold,
            )
            for idx, row, pred_words in zip(es_positions, es_rows, es_pred_words_list):
                predictions_by_idx[idx] = {"raw": list(row["raw"]), "pred": pred_words, "lang": "es"}
        else:
            print("Predicting ES with MFR...", flush=True)
            for idx, row in tqdm(zip(es_positions, es_rows), total=len(es_rows), desc="ES MFR"):
                raw_words = list(row["raw"])
                pred_words = predict_mfr(raw_words, mfr_by_lang.get("es", {}))
                predictions_by_idx[idx] = {"raw": raw_words, "pred": pred_words, "lang": "es"}

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
