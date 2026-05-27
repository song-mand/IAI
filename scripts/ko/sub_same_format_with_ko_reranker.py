# -*- coding: utf-8 -*-
"""Create submission.zip in the SAME format/order as the original sub.py.

This script intentionally follows the previous submission style:
  1. load_dataset("weerayut/multilexnorm2026-dev-pub")
  2. use train_split = full_dataset["train"] and eval_split = full_dataset["test"]
  3. iterate sorted(all_langs)
  4. filter train/eval by lang
  5. append {"raw": raw_words, "pred": pred_words, "lang": lang}

Only difference:
  - lang == "ko" uses the saved Korean LogisticRegression reranker.
  - If the reranker is missing/fails, it falls back to MFR.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import zipfile
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ko_reranker import load_bundle, predict_sentence


STRATEGY = {
    "da": "natural",
    "en": "natural",
    "sl": "natural",
    "sr": "natural",
    "hr": "natural",
    "iden": "natural",
    "de": "sentinel",
    "nl": "sentinel",
    "tr": "sentinel",
    "trde": "sentinel",
    "es": "mfr",
    "it": "mfr",
    "id": "mfr",
    "ja": "mfr",
    "ko": "ko_reranker",
    "th": "mfr",
    "vi": "mfr",
}

ALL_LANGS = sorted([
    "en", "da", "de", "es", "hr", "it", "nl", "sl", "sr", "tr",
    "iden", "trde", "id", "ja", "ko", "th", "vi",
])


def norm_or_raw(raw: str, norm: Any) -> str:
    """Treat empty/None/NaN labels as identity, same practical behavior as MFR fallback."""
    if norm is None:
        return raw
    try:
        if norm != norm:  # NaN
            return raw
    except Exception:
        pass
    if norm == "":
        return raw
    return str(norm)


def build_mfr_dictionary(train_data) -> Dict[str, str]:
    """Same shape as the original sub.py MFR dictionary builder."""
    mfr_counts: Dict[str, Dict[str, int]] = {}

    for row in train_data:
        for r, n in zip(row["raw"], row["norm"]):
            r = str(r)
            target = norm_or_raw(r, n)
            if r not in mfr_counts:
                mfr_counts[r] = {}
            mfr_counts[r][target] = mfr_counts[r].get(target, 0) + 1

    return {
        r: max(targets.items(), key=lambda x: (x[1], x[0] == r))[0]
        for r, targets in mfr_counts.items()
    }


def predict_with_mfr(raw_words: Sequence[Any], mfr_dict: Dict[str, str]) -> List[str]:
    return [mfr_dict.get(str(w), str(w)) for w in raw_words]


def predict_with_seq2seq(
    raw_words: Sequence[Any],
    lang: str,
    fmt: str,
    tokenizer: Any,
    model: Any,
    device: str,
) -> List[str]:
    raw_words = [str(w) for w in raw_words]
    context = " ".join(raw_words)
    inputs_list: List[str] = []

    for i, target_word in enumerate(raw_words):
        if fmt == "natural":
            inputs_list.append(f"lang: {lang} word: {target_word} context: {context}")
        else:
            words_copy = raw_words.copy()
            words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
            inputs_list.append(" ".join(words_copy))

    inputs = tokenizer(
        inputs_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=2)

    preds = [p.strip() for p in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    return preds if len(preds) == len(raw_words) else raw_words


def try_load_seq2seq(lang: str, model_root: str, device: str) -> Tuple[Any, Any]:
    model_path = os.path.join(model_root, f"{lang}_model")
    if not os.path.exists(model_path):
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"[{lang.upper()}] model load failed -> MFR fallback: {e}")
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--ko-model-dir", default="artifacts/ko_reranker")
    parser.add_argument("--seq2seq-model-root", default="final_model")
    parser.add_argument("--output-json", default="submission_files/predictions.json")
    parser.add_argument("--zip-path", default="submission.zip")
    parser.add_argument("--disable-ko-reranker", action="store_true")
    args = parser.parse_args()

    print("Loading dataset...")
    full_dataset = load_dataset(args.hf_dataset)
    train_split, eval_split = full_dataset["train"], full_dataset["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    all_predictions_for_json: List[dict] = []

    print("============================================================")
    print("Submission prediction: original sub.py order + KO reranker")
    print("============================================================")
    print(f"dataset          : {args.hf_dataset}")
    print(f"device           : {device}")
    print(f"ko model dir     : {args.ko_model_dir}")
    print(f"seq2seq root     : {args.seq2seq_model_root}")
    print("output order     : sorted language loop, same as old sub.py")
    print("============================================================")

    ko_bundle = None
    if not args.disable_ko_reranker:
        try:
            ko_bundle = load_bundle(args.ko_model_dir)
            print(f"[KO] reranker loaded: {args.ko_model_dir}")
        except Exception as e:
            print(f"[KO] reranker load failed -> MFR fallback: {e}")
            ko_bundle = None

    for lang in ALL_LANGS:
        lang_train = train_split.filter(lambda x: x["lang"] == lang)
        lang_eval = eval_split.filter(lambda x: x["lang"] == lang)
        if len(lang_eval) == 0:
            continue

        fmt = STRATEGY.get(lang, "mfr")
        use_deep_learning = fmt not in {"mfr", "ko_reranker"}
        tokenizer, model, mfr_dict = None, None, None

        if lang == "ko" and ko_bundle is not None:
            print(f"[{lang.upper()}] 예측 (KO reranker 적용)")
        else:
            if use_deep_learning:
                tokenizer, model = try_load_seq2seq(lang, args.seq2seq_model_root, device)
                if tokenizer is None or model is None:
                    use_deep_learning = False

            if not use_deep_learning:
                mfr_dict = build_mfr_dictionary(lang_train)
                if lang == "ko":
                    print(f"[{lang.upper()}] 예측 (MFR fallback 적용)")
                else:
                    print(f"[{lang.upper()}] 예측 (MFR 적용)")

        for row in tqdm(lang_eval, desc=f"[{lang.upper()}] 예측 ", leave=False):
            raw_words = row["raw"]

            if lang == "ko" and ko_bundle is not None:
                pred_words = predict_sentence([str(w) for w in raw_words], ko_bundle)
            elif use_deep_learning and tokenizer is not None and model is not None:
                pred_words = predict_with_seq2seq(raw_words, lang, fmt, tokenizer, model, device)
            else:
                pred_words = predict_with_mfr(raw_words, mfr_dict or {})

            if len(pred_words) != len(raw_words):
                pred_words = [str(w) for w in raw_words]

            # IMPORTANT: same JSON object shape as old sub.py
            all_predictions_for_json.append({
                "raw": raw_words,
                "pred": [str(w) for w in pred_words],
                "lang": lang,
            })

        if use_deep_learning and model is not None:
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_predictions_for_json, f, ensure_ascii=False)

    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(args.output_json, arcname="predictions.json")

    print("============================================================")
    print("압축 완료")
    print(f"json: {args.output_json}")
    print(f"zip : {args.zip_path}")
    print(f"rows: {len(all_predictions_for_json)}")
    print("============================================================")


if __name__ == "__main__":
    main()
