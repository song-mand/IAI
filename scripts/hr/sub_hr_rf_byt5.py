#!/usr/bin/env python3
"""
Create submission.zip with this policy:
- Croatian(hr): RandomForest decides whether each token should be changed.
  Only tokens predicted as changed are sent to the HR ByT5 model.
- Every non-HR language: MFR only.

The output order follows the original dataset test split order to avoid raw-order
mismatch during scoring.

This version prints diagnostic statistics that are useful for debugging:
- per-language sentence/token/change counts
- HR RF gate candidate counts and probability summary
- HR ByT5 output-change and empty-output counts
- a small sample of HR gated examples
"""

import argparse
import gc
import json
import os
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from joblib import load
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG = "hr"


def build_mfr_dictionary(rows: Sequence[dict]) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = raw if norm is None else norm
            counts[raw][target] += 1
    return {
        raw: max(targets.items(), key=lambda item: (item[1], item[0] == raw))[0]
        for raw, targets in counts.items()
    }


def _safe_token(tokens: Sequence[str], idx: int) -> str:
    if 0 <= idx < len(tokens):
        tok = tokens[idx]
        return "" if tok is None else str(tok)
    return "<BOS>" if idx < 0 else "<EOS>"


def token_features(raw_words: Sequence[str], i: int, mfr_dict: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    word = _safe_token(raw_words, i)
    lower = word.lower()
    prev1 = _safe_token(raw_words, i - 1).lower()
    prev2 = _safe_token(raw_words, i - 2).lower()
    next1 = _safe_token(raw_words, i + 1).lower()
    next2 = _safe_token(raw_words, i + 2).lower()

    n_chars = max(len(word), 1)
    n_alpha = sum(ch.isalpha() for ch in word)
    n_digit = sum(ch.isdigit() for ch in word)
    n_upper = sum(ch.isupper() for ch in word)
    n_punct = sum((not ch.isalnum()) for ch in word)
    repeated_char = any(word[j] == word[j - 1] for j in range(1, len(word)))
    mfr_pred = mfr_dict.get(word, word) if mfr_dict is not None else word

    feats: Dict[str, object] = {
        "bias": 1,
        "tok=" + lower: 1,
        "prev1=" + prev1: 1,
        "prev2=" + prev2: 1,
        "next1=" + next1: 1,
        "next2=" + next2: 1,
        "prev_bigram=" + prev1 + "|" + lower: 1,
        "next_bigram=" + lower + "|" + next1: 1,
        "len": len(word),
        "sent_len": len(raw_words),
        "pos_abs": i,
        "pos_rel": i / max(len(raw_words) - 1, 1),
        "is_lower": int(word.islower()),
        "is_upper": int(word.isupper()),
        "is_title": int(word.istitle()),
        "has_alpha": int(n_alpha > 0),
        "has_digit": int(n_digit > 0),
        "has_punct": int(n_punct > 0),
        "has_at": int("@" in word),
        "has_hash": int("#" in word),
        "has_url_piece": int("http" in lower or "www" in lower or ".com" in lower),
        "repeated_char": int(repeated_char),
        "digit_ratio": n_digit / n_chars,
        "upper_ratio": n_upper / n_chars,
        "punct_ratio": n_punct / n_chars,
        "alpha_ratio": n_alpha / n_chars,
        "mfr_seen": int(mfr_dict is not None and word in mfr_dict),
        "mfr_would_change": int(mfr_pred != word),
        "mfr_pred=" + mfr_pred.lower(): 1,
    }
    for k in range(1, 5):
        if len(lower) >= k:
            feats[f"pref{k}=" + lower[:k]] = 1
            feats[f"suf{k}=" + lower[-k:]] = 1
    return feats


def batched(items: Sequence[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def predict_hr_sentence(
    raw_words: Sequence[str],
    rf_pipeline,
    threshold: float,
    tokenizer,
    byt5_model,
    device: str,
    mfr_dict: Dict[str, str],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    want_examples: int = 0,
) -> Tuple[List[str], Dict[str, object]]:
    pred_words = list(raw_words)
    features = [token_features(raw_words, i, mfr_dict) for i in range(len(raw_words))]
    change_proba = rf_pipeline.predict_proba(features)[:, 1]
    change_indices = [i for i, p in enumerate(change_proba) if float(p) >= threshold]

    debug: Dict[str, object] = {
        "tokens": len(raw_words),
        "rf_candidates": len(change_indices),
        "rf_proba_sum": float(sum(float(p) for p in change_proba)),
        "rf_proba_min": float(min(change_proba)) if len(change_proba) else 0.0,
        "rf_proba_max": float(max(change_proba)) if len(change_proba) else 0.0,
        "byt5_called": 0,
        "byt5_empty": 0,
        "byt5_output_diff_raw": 0,
        "final_changed": 0,
        "examples": [],
    }

    if not change_indices:
        return pred_words, debug

    context = " ".join(raw_words)
    inputs_for_model = [
        f"lang: {LANG} word: {raw_words[i]} context: {context}" for i in change_indices
    ]
    debug["byt5_called"] = len(inputs_for_model)

    decoded_all: List[str] = []
    for _, batch_inputs in batched(inputs_for_model, batch_size):
        encoded = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        with torch.no_grad():
            outputs = byt5_model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_all.extend([text.strip() for text in decoded])

    examples = []
    for idx, decoded in zip(change_indices, decoded_all):
        if not decoded:
            debug["byt5_empty"] = int(debug["byt5_empty"]) + 1
            final = raw_words[idx]
        else:
            final = decoded
            if decoded != raw_words[idx]:
                debug["byt5_output_diff_raw"] = int(debug["byt5_output_diff_raw"]) + 1

        pred_words[idx] = final

        if want_examples > 0 and len(examples) < want_examples:
            left = " ".join(raw_words[max(0, idx - 3):idx])
            right = " ".join(raw_words[idx + 1:idx + 4])
            examples.append({
                "idx": idx,
                "raw": raw_words[idx],
                "pred": final,
                "rf_p": round(float(change_proba[idx]), 4),
                "context": f"{left} >>> {raw_words[idx]} <<< {right}".strip(),
            })

    debug["final_changed"] = sum(1 for raw, pred in zip(raw_words, pred_words) if raw != pred)
    debug["examples"] = examples
    return pred_words, debug


def pct(num: int, den: int) -> str:
    return "0.00%" if den == 0 else f"{num / den * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--work_dir", default=".")
    parser.add_argument("--model_dir", default="final_model")
    parser.add_argument("--rf_path", default=None)
    parser.add_argument("--byt5_path", default=None)
    parser.add_argument("--output_dir", default="submission_files")
    parser.add_argument("--output_zip", default="submission.zip")
    parser.add_argument("--force_mfr_hr", action="store_true")
    parser.add_argument("--hr_threshold", type=float, default=None)
    parser.add_argument("--byt5_batch_size", type=int, default=64)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--debug_examples", type=int, default=8,
                        help="Number of HR gated examples to print/save for debugging. Set 0 to disable.")
    parser.add_argument("--debug_json", default=None,
                        help="Optional path to save diagnostic stats as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = os.path.join(args.work_dir, args.model_dir)
    rf_path = args.rf_path or os.path.join(model_dir, "hr_change_rf.joblib")
    byt5_path = args.byt5_path or os.path.join(model_dir, "hr_model")

    print("==============================")
    print("1. Load dataset")
    print("==============================")
    full_dataset = load_dataset(args.dataset)
    train_split = full_dataset[args.train_split]
    eval_split = full_dataset[args.eval_split]
    train_lang_counts = defaultdict(int)
    eval_lang_counts = defaultdict(int)
    for row in train_split:
        train_lang_counts[row["lang"]] += 1
    for row in eval_split:
        eval_lang_counts[row["lang"]] += 1
    print(f"train sentences: {len(train_split):,} | eval sentences: {len(eval_split):,}")
    print("eval sentence counts by lang:", dict(sorted(eval_lang_counts.items())))

    print("==============================")
    print("2. Build MFR dictionaries for all languages")
    print("==============================")
    langs = sorted(set(train_split["lang"]))
    mfr_by_lang: Dict[str, Dict[str, str]] = {}
    for lang in langs:
        lang_rows = [row for row in train_split if row["lang"] == lang]
        mfr_by_lang[lang] = build_mfr_dictionary(lang_rows)
        print(f"[{lang.upper()}] MFR entries: {len(mfr_by_lang[lang]):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_hr_logic = False
    rf_pipeline = None
    threshold = None
    tokenizer = None
    byt5_model = None

    if not args.force_mfr_hr and os.path.exists(rf_path) and os.path.exists(byt5_path):
        print("==============================")
        print("3. Load HR RF gate + ByT5 model")
        print("==============================")
        bundle = load(rf_path)
        rf_pipeline = bundle["pipeline"]
        threshold = args.hr_threshold if args.hr_threshold is not None else float(bundle.get("threshold", 0.5))
        tokenizer = AutoTokenizer.from_pretrained(byt5_path)
        byt5_model = AutoModelForSeq2SeqLM.from_pretrained(byt5_path).to(device)
        byt5_model.eval()
        use_hr_logic = True
        print(f"HR logic enabled | threshold={threshold:.4f} | device={device}")
        print(f"RF path: {rf_path}")
        print(f"ByT5 path: {byt5_path}")
    else:
        print("HR RF/ByT5 files not found or --force_mfr_hr set. HR also falls back to MFR.")
        print(f"checked RF path: {rf_path}")
        print(f"checked ByT5 path: {byt5_path}")

    print("==============================")
    print("4. Predict in original eval order")
    print("==============================")
    os.makedirs(args.output_dir, exist_ok=True)
    all_predictions = []

    lang_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hr_debug_totals: Dict[str, object] = {
        "sentences": 0,
        "tokens": 0,
        "rf_candidates": 0,
        "rf_proba_sum": 0.0,
        "rf_proba_min": None,
        "rf_proba_max": None,
        "byt5_called": 0,
        "byt5_empty": 0,
        "byt5_output_diff_raw": 0,
        "final_changed": 0,
        "examples": [],
    }

    for row in tqdm(eval_split, desc="Predict"):
        lang = row["lang"]
        raw_words = row["raw"]
        lang_stats[lang]["sentences"] += 1
        lang_stats[lang]["tokens"] += len(raw_words)

        if lang == LANG and use_hr_logic:
            remaining_examples = max(0, args.debug_examples - len(hr_debug_totals["examples"]))
            pred_words, debug = predict_hr_sentence(
                raw_words=raw_words,
                rf_pipeline=rf_pipeline,
                threshold=threshold,
                tokenizer=tokenizer,
                byt5_model=byt5_model,
                device=device,
                mfr_dict=mfr_by_lang.get(lang, {}),
                batch_size=args.byt5_batch_size,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                want_examples=remaining_examples,
            )
            hr_debug_totals["sentences"] = int(hr_debug_totals["sentences"]) + 1
            for key in ["tokens", "rf_candidates", "byt5_called", "byt5_empty", "byt5_output_diff_raw", "final_changed"]:
                hr_debug_totals[key] = int(hr_debug_totals[key]) + int(debug[key])
            hr_debug_totals["rf_proba_sum"] = float(hr_debug_totals["rf_proba_sum"]) + float(debug["rf_proba_sum"])
            if debug["tokens"]:
                cur_min = debug["rf_proba_min"]
                cur_max = debug["rf_proba_max"]
                old_min = hr_debug_totals["rf_proba_min"]
                old_max = hr_debug_totals["rf_proba_max"]
                hr_debug_totals["rf_proba_min"] = cur_min if old_min is None else min(float(old_min), float(cur_min))
                hr_debug_totals["rf_proba_max"] = cur_max if old_max is None else max(float(old_max), float(cur_max))
            if len(hr_debug_totals["examples"]) < args.debug_examples:
                hr_debug_totals["examples"].extend(debug.get("examples", []))
                hr_debug_totals["examples"] = hr_debug_totals["examples"][:args.debug_examples]
        else:
            mfr_dict = mfr_by_lang.get(lang, {})
            pred_words = [mfr_dict.get(word, word) for word in raw_words]
            if lang == LANG:
                lang_stats[lang]["hr_fallback_mfr_sentences"] += 1

        # Hard guard for Codabench/scoring compatibility.
        if len(pred_words) != len(raw_words):
            raise RuntimeError(
                f"Prediction length mismatch for lang={lang}: raw={len(raw_words)} pred={len(pred_words)}"
            )

        changed = sum(1 for raw, pred in zip(raw_words, pred_words) if raw != pred)
        lang_stats[lang]["changed_tokens"] += changed
        all_predictions.append({"raw": raw_words, "pred": pred_words, "lang": lang})

    print("==============================")
    print("5. Prediction summary")
    print("==============================")
    for lang in sorted(lang_stats):
        st = lang_stats[lang]
        print(
            f"[{lang.upper()}] sentences={st['sentences']:,} "
            f"tokens={st['tokens']:,} "
            f"pred_changed={st['changed_tokens']:,} ({pct(st['changed_tokens'], st['tokens'])})"
        )
        if lang == LANG and st.get("hr_fallback_mfr_sentences", 0):
            print(f"  HR fallback MFR sentences: {st['hr_fallback_mfr_sentences']:,}")

    if use_hr_logic:
        print("==============================")
        print("6. HR RF gate / ByT5 diagnostics")
        print("==============================")
        hr_tokens = int(hr_debug_totals["tokens"])
        rf_candidates = int(hr_debug_totals["rf_candidates"])
        avg_p = float(hr_debug_totals["rf_proba_sum"]) / hr_tokens if hr_tokens else 0.0
        print(f"HR sentences: {hr_debug_totals['sentences']:,}")
        print(f"HR tokens: {hr_tokens:,}")
        print(f"RF threshold: {threshold:.4f}")
        print(f"RF candidates sent to ByT5: {rf_candidates:,} ({pct(rf_candidates, hr_tokens)})")
        print(
            "RF probability: "
            f"avg={avg_p:.4f}, "
            f"min={float(hr_debug_totals['rf_proba_min'] or 0.0):.4f}, "
            f"max={float(hr_debug_totals['rf_proba_max'] or 0.0):.4f}"
        )
        print(f"ByT5 calls: {hr_debug_totals['byt5_called']:,}")
        print(f"ByT5 empty outputs: {hr_debug_totals['byt5_empty']:,}")
        print(
            f"ByT5 outputs different from raw: {hr_debug_totals['byt5_output_diff_raw']:,} "
            f"({pct(int(hr_debug_totals['byt5_output_diff_raw']), int(hr_debug_totals['byt5_called']))})"
        )
        print(
            f"Final HR changed tokens: {hr_debug_totals['final_changed']:,} "
            f"({pct(int(hr_debug_totals['final_changed']), hr_tokens)})"
        )

        if args.debug_examples > 0 and hr_debug_totals["examples"]:
            print("------------------------------")
            print("HR gated examples")
            print("------------------------------")
            for ex in hr_debug_totals["examples"]:
                print(f"p={ex['rf_p']:.4f} | {ex['raw']} -> {ex['pred']} | {ex['context']}")

    json_path = os.path.join(args.output_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)

    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")

    if args.debug_json:
        debug_payload = {
            "use_hr_logic": use_hr_logic,
            "threshold": threshold,
            "lang_stats": {lang: dict(st) for lang, st in lang_stats.items()},
            "hr_debug": hr_debug_totals,
        }
        with open(args.debug_json, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)
        print(f"Saved debug JSON: {args.debug_json}")

    print(f"Saved: {json_path}")
    print(f"Created: {args.output_zip}")

    if byt5_model is not None:
        del byt5_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
