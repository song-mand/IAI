# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from th_utils import (
    build_mfr_for_any_lang,
    build_resources,
    edit_distance,
    evaluate_token_level,
    extract_features,
    get_mfr_info,
    has_thai,
    is_protected_token,
    load_pickle,
    save_pickle,
    seed_everything,
    simple_non_byt5_candidate,
    target_or_raw,
    thai_normalize_token,
    thai_rule_candidates,
)

ALL_LANGS_DEFAULT = "en,da,de,es,hr,it,nl,sl,sr,tr,iden,trde,id,ja,ko,th,vi"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("DATASET", "weerayut/multilexnorm2026-dev-pub"))
    p.add_argument("--train-split", default=os.environ.get("TRAIN_SPLIT", "train"))
    p.add_argument("--eval-split", default=os.environ.get("EVAL_SPLIT", "test"))
    p.add_argument("--all-langs", default=os.environ.get("ALL_LANGS", ALL_LANGS_DEFAULT))
    p.add_argument("--lang", default=os.environ.get("LANG", "th"))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))

    # Paths
    p.add_argument("--resource-path", default=os.environ.get("TH_RESOURCE_PATH", "models/th/th_resources.pkl"))
    p.add_argument("--detector-path", default=os.environ.get("TH_DETECTOR_PATH", "models/th/th_detector.joblib"))
    p.add_argument("--byt5-model-path", default=os.environ.get("TH_BYT5_MODEL", "final_model/th_byt5_candidate"))
    p.add_argument("--submission-dir", default=os.environ.get("SUBMISSION_DIR", "submission_files"))
    p.add_argument("--zip-path", default=os.environ.get("ZIP_PATH", "submission.zip"))
    p.add_argument("--debug-path", default=os.environ.get("DEBUG_PATH", "submission_files/th_debug_samples.json"))

    # Main switches
    p.add_argument("--use-detector", type=int, default=int(os.environ.get("USE_DETECTOR", "1")))
    p.add_argument("--use-byt5", type=int, default=int(os.environ.get("USE_BYT5", "1")))
    p.add_argument("--use-rules", type=int, default=int(os.environ.get("USE_RULES", "1")))
    p.add_argument("--use-errdict", type=int, default=int(os.environ.get("USE_ERRDICT", "1")))
    p.add_argument("--use-mfr-candidate", type=int, default=int(os.environ.get("USE_MFR_CANDIDATE", "1")))

    # Thresholds / reranking
    p.add_argument("--detector-threshold", default=os.environ.get("DETECTOR_THRESHOLD", "auto"))
    p.add_argument("--byt5-threshold", type=float, default=float(os.environ.get("BYT5_THRESHOLD", "0.55")))
    p.add_argument("--min-mfr-conf", type=float, default=float(os.environ.get("MIN_MFR_CONF", "0.80")))
    p.add_argument("--force-mfr-conf", type=float, default=float(os.environ.get("FORCE_MFR_CONF", "0.93")))
    p.add_argument("--min-mfr-count", type=int, default=int(os.environ.get("MIN_MFR_COUNT", "2")))
    p.add_argument("--accept-score-min", type=float, default=float(os.environ.get("ACCEPT_SCORE_MIN", "2.40")))
    p.add_argument("--max-edit-ratio", type=float, default=float(os.environ.get("MAX_EDIT_RATIO", "0.75")))
    p.add_argument("--byt5-bonus", type=float, default=float(os.environ.get("BYT5_BONUS", "1.20")))
    p.add_argument("--require-correct-dict-for-byt5", type=int, default=int(os.environ.get("REQUIRE_CORRECT_DICT_FOR_BYT5", "0")))
    p.add_argument("--protect-nonthai", type=int, default=int(os.environ.get("PROTECT_NONTHAI", "1")))
    p.add_argument("--max-rule-candidates", type=int, default=int(os.environ.get("MAX_RULE_CANDIDATES", "8")))
    p.add_argument("--max-errdict-candidates", type=int, default=int(os.environ.get("MAX_ERRDICT_CANDIDATES", "5")))

    # ByT5 generation
    p.add_argument("--input-format", default=os.environ.get("INPUT_FORMAT", "natural"), choices=["natural", "sentinel"])
    p.add_argument("--max-input-length", type=int, default=int(os.environ.get("MAX_INPUT_LENGTH", "160")))
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "32")))
    p.add_argument("--num-beams", type=int, default=int(os.environ.get("NUM_BEAMS", "2")))
    p.add_argument("--num-return-sequences", type=int, default=int(os.environ.get("NUM_RETURN_SEQUENCES", "1")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("INFER_BATCH_SIZE", "64")))

    # Debug prints
    p.add_argument("--print-examples", type=int, default=int(os.environ.get("PRINT_EXAMPLES", "80")))
    p.add_argument("--print-lang-metrics", type=int, default=int(os.environ.get("PRINT_LANG_METRICS", "1")))
    return p.parse_args()


def make_byt5_input(lang: str, word: str, raw_words: List[str], index: int, fmt: str) -> str:
    if fmt == "sentinel":
        copied = list(raw_words)
        copied[index] = f"<extra_id_0> {word} <extra_id_1>"
        return " ".join(copied)
    return f"lexnorm lang: {lang} word: {word} context: {' '.join(raw_words)}"


def load_detector(path: str):
    if not os.path.exists(path):
        print(f"[WARN] detector not found: {path}")
        return None, None
    import joblib

    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        threshold = payload.get("threshold")
        print(f"loaded detector: {path} threshold={threshold}")
        return payload["model"], threshold
    print(f"loaded raw detector: {path}")
    return payload, None


def resolve_threshold(arg_value: str, trained_threshold: Optional[float]) -> float:
    if str(arg_value).lower() == "auto":
        if trained_threshold is not None:
            return float(trained_threshold)
        return 0.70
    return float(arg_value)


def load_or_build_resources(args, train_rows):
    if os.path.exists(args.resource_path):
        print(f"loaded resources: {args.resource_path}")
        return load_pickle(args.resource_path)
    print(f"[WARN] resources not found; rebuilding from train split: {args.resource_path}")
    rows = [r for r in train_rows if r.get("lang") == args.lang]
    res = build_resources(rows, lang=args.lang)
    save_pickle(res, args.resource_path)
    return res


def load_byt5(args, device):
    if not args.use_byt5:
        return None, None
    if not os.path.exists(args.byt5_model_path):
        print(f"[WARN] ByT5 model not found, skip ByT5 candidates: {args.byt5_model_path}")
        return None, None
    print(f"loading ByT5 candidate model: {args.byt5_model_path}")
    tok = AutoTokenizer.from_pretrained(args.byt5_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.byt5_model_path).to(device)
    model.eval()
    return tok, model


def generate_byt5_candidates(tokenizer, model, inputs: List[str], args, device) -> List[List[str]]:
    if not inputs:
        return []
    grouped: List[List[str]] = []
    nret = max(1, args.num_return_sequences)
    for start in range(0, len(inputs), args.batch_size):
        batch = inputs[start : start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                num_beams=max(args.num_beams, nret),
                num_return_sequences=nret,
                do_sample=False,
            )
        dec = [x.strip() for x in tokenizer.batch_decode(out, skip_special_tokens=True)]
        if nret == 1:
            grouped.extend([[x] for x in dec])
        else:
            for i in range(0, len(dec), nret):
                grouped.append(dec[i : i + nret])
    return grouped


def candidate_valid(raw: str, cand: str, source: str, resources: Dict[str, Any], args) -> Tuple[bool, str]:
    raw = str(raw)
    cand = "" if cand is None else str(cand).strip()
    if not cand:
        return False, "empty"
    if cand == raw:
        return False, "same"
    if any(ch.isspace() for ch in cand):
        return False, "space"
    if len(cand) > max(2, len(raw) * 2 + 6):
        return False, "too_long"
    if has_thai(raw) and not has_thai(cand):
        return False, "lost_thai"
    if args.protect_nonthai and is_protected_token(raw) and source not in {"mfr", "errdict"}:
        return False, "protected"

    dist = edit_distance(thai_normalize_token(raw), thai_normalize_token(cand), max_distance=20)
    ratio = dist / max(1, max(len(raw), len(cand)))
    if ratio > args.max_edit_ratio and source not in {"mfr", "errdict"}:
        return False, "far_edit"

    if source == "byt5" and args.require_correct_dict_for_byt5:
        if cand not in resources.get("correct_set", set()):
            return False, "byt5_not_correct_dict"
    return True, "ok"


def score_candidate(raw: str, cand: str, source: str, detector_prob: float, resources: Dict[str, Any], args) -> float:
    correct_set = resources.get("correct_set", set())
    mfr_best, st = get_mfr_info(resources, raw)
    dist = edit_distance(thai_normalize_token(raw), thai_normalize_token(cand), max_distance=20)
    score = 0.8 * detector_prob
    score += 0.7 if cand in correct_set else 0.0
    score += 0.8 if cand == mfr_best and cand != raw else 0.0
    score -= 0.10 * dist
    score -= 0.05 * abs(len(cand) - len(raw))

    if source == "mfr":
        score += 2.2 + 2.5 * st.get("conf", 0.0) + math.log1p(st.get("total", 0.0)) / 5.0
    elif source == "errdict":
        err_map = resources.get("error_dict", {}).get(raw, {})
        cnt = err_map.get(cand, 0)
        total = sum(err_map.values())
        score += 2.0 + 2.0 * (cnt / max(1, total)) + math.log1p(cnt) / 5.0
    elif source == "rule":
        score += 1.35 + (0.8 if cand in correct_set else 0.0)
    elif source == "byt5":
        score += args.byt5_bonus + (1.2 if cand in correct_set else 0.0) + (1.0 if cand == mfr_best and cand != raw else 0.0)
    return float(score)


def add_candidate(cands: List[Tuple[str, str, float]], raw, cand, source, detector_prob, resources, args, reject_counter):
    ok, reason = candidate_valid(raw, cand, source, resources, args)
    if not ok:
        reject_counter[reason] += 1
        return
    score = score_candidate(raw, cand, source, detector_prob, resources, args)
    cands.append((str(cand), source, score))


def predict_th_sentence(raw_words, resources, detector, tokenizer, model, args, device, stats):
    sent_len = len(raw_words)
    probs: List[float] = []
    mfr_infos: List[Tuple[str, Dict[str, float]]] = []

    # 1) detector probabilities
    for i, raw in enumerate(raw_words):
        prev_tok = raw_words[i - 1] if i > 0 else ""
        next_tok = raw_words[i + 1] if i + 1 < sent_len else ""
        if args.use_detector and detector is not None:
            feat = [extract_features(raw, prev_tok, next_tok, i, sent_len, resources)]
            prob = float(detector.predict_proba(feat)[0][1])
        else:
            cand, src, score = simple_non_byt5_candidate(raw, resources, args.min_mfr_conf, args.min_mfr_count)
            prob = 0.90 if cand != raw else 0.05
        probs.append(prob)
        mfr_infos.append(get_mfr_info(resources, raw))

    # 2) ByT5 candidate generation only for likely changed tokens.
    byt5_inputs = []
    byt5_positions = []
    if tokenizer is not None and model is not None:
        for i, raw in enumerate(raw_words):
            mfr_best, st = mfr_infos[i]
            force_mfr = mfr_best != raw and st.get("conf", 0.0) >= args.force_mfr_conf and st.get("total", 0.0) >= args.min_mfr_count
            if probs[i] >= args.byt5_threshold or force_mfr or raw in resources.get("error_dict", {}):
                byt5_positions.append(i)
                byt5_inputs.append(make_byt5_input(args.lang, raw, raw_words, i, args.input_format))
    byt5_grouped = generate_byt5_candidates(tokenizer, model, byt5_inputs, args, device) if byt5_inputs else []
    byt5_by_pos = {pos: cands for pos, cands in zip(byt5_positions, byt5_grouped)}
    stats["byt5_called_tokens"] += len(byt5_positions)

    # 3) candidate collection and reranking
    pred = []
    decisions = []
    for i, raw in enumerate(raw_words):
        raw = str(raw)
        prob = probs[i]
        mfr_best, st = mfr_infos[i]
        force_mfr = mfr_best != raw and st.get("conf", 0.0) >= args.force_mfr_conf and st.get("total", 0.0) >= args.min_mfr_count
        allow_change = prob >= args.detector_threshold_resolved or force_mfr

        reject_counter = stats["reject_reasons"]
        cands: List[Tuple[str, str, float]] = []

        if args.use_mfr_candidate and mfr_best != raw and st.get("conf", 0.0) >= args.min_mfr_conf and st.get("total", 0.0) >= args.min_mfr_count:
            add_candidate(cands, raw, mfr_best, "mfr", prob, resources, args, reject_counter)

        if args.use_errdict:
            err_map = resources.get("error_dict", {}).get(raw, {})
            for cand, _ in sorted(err_map.items(), key=lambda x: (-x[1], x[0]))[: args.max_errdict_candidates]:
                add_candidate(cands, raw, cand, "errdict", prob, resources, args, reject_counter)

        if args.use_rules:
            for cand in thai_rule_candidates(raw, max_candidates=args.max_rule_candidates):
                add_candidate(cands, raw, cand, "rule", prob, resources, args, reject_counter)

        for cand in byt5_by_pos.get(i, []):
            add_candidate(cands, raw, cand, "byt5", prob, resources, args, reject_counter)

        # Deduplicate by candidate, keeping max score.
        best_by_cand: Dict[str, Tuple[str, str, float]] = {}
        for cand, src, score in cands:
            if cand not in best_by_cand or score > best_by_cand[cand][2]:
                best_by_cand[cand] = (cand, src, score)
        cands = list(best_by_cand.values())
        cands.sort(key=lambda x: x[2], reverse=True)

        if allow_change and cands and cands[0][2] >= args.accept_score_min:
            cand, src, score = cands[0]
            pred.append(cand)
            stats["source_counter"][src] += 1
            stats["changed_tokens"] += 1
            decisions.append({"i": i, "raw": raw, "pred": cand, "source": src, "prob": prob, "score": score, "top_candidates": cands[:4]})
        else:
            pred.append(raw)
            stats["source_counter"]["keep"] += 1
            if cands:
                stats["blocked_with_candidates"] += 1

        stats["prob_values"].append(prob)
        stats["candidate_count_total"] += len(cands)
    return pred, decisions


def mfr_predict_sentence(raw_words, mfr_dict):
    return [mfr_dict.get(str(w), str(w)) for w in raw_words]


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.submission_dir, exist_ok=True)
    print("=" * 80)
    print("Thai two-stage submission")
    print("=" * 80)
    print(vars(args))

    full = load_dataset(args.dataset)
    train_split = full[args.train_split]
    eval_split = full[args.eval_split]
    all_langs = [x.strip() for x in args.all_langs.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    train_rows = list(train_split)
    eval_rows = list(eval_split)
    resources = load_or_build_resources(args, train_rows)
    detector, trained_th = load_detector(args.detector_path) if args.use_detector else (None, None)
    args.detector_threshold_resolved = resolve_threshold(args.detector_threshold, trained_th)
    print(f"detector_threshold_resolved={args.detector_threshold_resolved:.4f}")
    print(f"TH resources: rows={resources.get('n_rows')} tokens={resources.get('n_tokens')} changed={resources.get('n_changed')} unique_raw={len(resources.get('raw_counter',{}))} error_keys={len(resources.get('error_dict',{}))}")

    tokenizer, byt5_model = load_byt5(args, device)

    # MFR dictionaries for non-Thai languages.
    mfr_by_lang = {}
    for lang in all_langs:
        if lang == args.lang:
            continue
        mfr_by_lang[lang] = build_mfr_for_any_lang(train_rows, lang)

    all_predictions = []
    debug_samples = []
    stats = {
        "source_counter": Counter(),
        "reject_reasons": Counter(),
        "changed_tokens": 0,
        "byt5_called_tokens": 0,
        "blocked_with_candidates": 0,
        "prob_values": [],
        "candidate_count_total": 0,
    }

    # For public dev-pub, eval_split may contain norm. Never uses norm for prediction;
    # only uses it after prediction for analysis prints.
    raw_gold_by_lang = defaultdict(lambda: {"raw": [], "gold": [], "pred": [], "mfr_pred": []})

    print("-" * 80)
    print("Predicting...")
    for row in tqdm(eval_rows, desc="all languages"):
        lang = row.get("lang")
        raw_words = [str(x) for x in row.get("raw", [])]
        if lang == args.lang:
            pred_words, decisions = predict_th_sentence(raw_words, resources, detector, tokenizer, byt5_model, args, device, stats)
            if len(debug_samples) < args.print_examples and decisions:
                debug_samples.append({"raw": raw_words, "pred": pred_words, "decisions": decisions})
        else:
            pred_words = mfr_predict_sentence(raw_words, mfr_by_lang.get(lang, {}))

        all_predictions.append({"raw": raw_words, "pred": pred_words, "lang": lang})

        if args.print_lang_metrics and row.get("norm") is not None:
            gold = [target_or_raw(r, n) for r, n in zip(raw_words, row.get("norm"))]
            raw_gold_by_lang[lang]["raw"].append(raw_words)
            raw_gold_by_lang[lang]["gold"].append(gold)
            raw_gold_by_lang[lang]["pred"].append(pred_words)
            if lang == args.lang:
                mfr_dict_th = resources.get("mfr", {})
                raw_gold_by_lang[lang]["mfr_pred"].append(mfr_predict_sentence(raw_words, mfr_dict_th))

    # Save predictions.
    json_path = os.path.join(args.submission_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")

    # Debug output.
    debug_obj = {
        "args": vars(args),
        "stats": {
            "source_counter": dict(stats["source_counter"]),
            "reject_reasons": dict(stats["reject_reasons"]),
            "changed_tokens": stats["changed_tokens"],
            "byt5_called_tokens": stats["byt5_called_tokens"],
            "blocked_with_candidates": stats["blocked_with_candidates"],
            "candidate_count_total": stats["candidate_count_total"],
        },
        "samples": debug_samples,
    }
    os.makedirs(os.path.dirname(args.debug_path) or ".", exist_ok=True)
    with open(args.debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Submission complete")
    print("=" * 80)
    print(f"predictions: {json_path}")
    print(f"zip:         {args.zip_path}")
    print(f"debug:       {args.debug_path}")
    print("-" * 80)
    print("TH decision stats")
    print(f"source_counter={dict(stats['source_counter'])}")
    print(f"reject_reasons={dict(stats['reject_reasons'])}")
    print(f"changed_tokens={stats['changed_tokens']}")
    print(f"byt5_called_tokens={stats['byt5_called_tokens']}")
    print(f"blocked_with_candidates={stats['blocked_with_candidates']}")
    print(f"candidate_count_total={stats['candidate_count_total']}")
    if stats["prob_values"]:
        probs = sorted(stats["prob_values"])
        def q(p):
            return probs[min(len(probs)-1, max(0, int((len(probs)-1)*p)))]
        print(f"detector prob quantiles: min={probs[0]:.3f} q25={q(.25):.3f} q50={q(.50):.3f} q75={q(.75):.3f} q90={q(.90):.3f} max={probs[-1]:.3f}")

    if args.print_lang_metrics:
        print("-" * 80)
        print("Metrics on eval split only if gold norm is available. Official hidden test usually has no norm.")
        for lang in all_langs:
            data = raw_gold_by_lang.get(lang)
            if not data or not data["raw"]:
                continue
            if lang == args.lang and data["mfr_pred"]:
                print(f"[{lang.upper()}] MFR baseline")
                evaluate_token_level(data["raw"], data["gold"], data["mfr_pred"], info=True, prefix="  ")
            print(f"[{lang.upper()}] submission pred")
            evaluate_token_level(data["raw"], data["gold"], data["pred"], info=True, prefix="  ")

    print("-" * 80)
    print("Sample TH modifications")
    shown = 0
    for sample in debug_samples:
        for d in sample["decisions"]:
            print(f"  {d['raw']} -> {d['pred']}  src={d['source']} prob={d['prob']:.3f} score={d['score']:.2f}")
            shown += 1
            if shown >= min(args.print_examples, 40):
                break
        if shown >= min(args.print_examples, 40):
            break

    if byt5_model is not None:
        del byt5_model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
