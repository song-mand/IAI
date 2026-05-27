# -*- coding: utf-8 -*-
"""Create an all-language submission while using the Korean contextual reranker.

Important:
  The official/local scorer checks raw rows exactly:
      assert label['raw'].tolist() == pred['raw'].tolist()

Therefore this script always writes predictions in the SAME ROW ORDER as the
input --test-path. For local validation scoring, pass the validation parquet as
--test-path. For final Codabench submission, pass the real test parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

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


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "tolist"):
        y = x.tolist()
        return y if isinstance(y, list) else [y]
    return list(x)


def jsonable_scalar(x: Any) -> Any:
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    # JSON cannot serialize some scalar objects. Converting unknown objects to str
    # is safer for token data, but normal Python str/int values are preserved.
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def jsonable_list(x: Any) -> List[Any]:
    return [jsonable_scalar(v) for v in as_list(x)]


def raw_to_model_words(raw: Any) -> List[str]:
    return [str(x) for x in jsonable_list(raw)]


def norm_or_raw(raw: str, norm: Any) -> str:
    if norm is None:
        return raw
    try:
        if norm != norm:  # NaN check
            return raw
    except Exception:
        pass
    if norm == "":
        return raw
    return str(norm)


def load_split(path: str | None, hf_dataset: str, split: str) -> List[dict]:
    if path:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")

    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    return [dict(row) for row in ds]


def filter_lang(rows: Sequence[dict], lang: str) -> List[dict]:
    return [row for row in rows if row.get("lang") == lang]


def build_mfr_dictionary(rows: Iterable[dict]) -> Dict[str, str]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        raw_words = raw_to_model_words(row.get("raw"))
        norm_words = as_list(row.get("norm"))
        if not norm_words:
            continue
        for raw, norm in zip(raw_words, norm_words):
            target = norm_or_raw(raw, norm)
            counts[raw][target] += 1

    # Tie-break: prefer non-identity if frequency is tied.
    return {
        raw: max(targets.items(), key=lambda x: (x[1], x[0] != raw))[0]
        for raw, targets in counts.items()
    }


def predict_with_mfr(raw_words: Sequence[str], mfr_dict: Dict[str, str]) -> List[str]:
    return [mfr_dict.get(w, w) for w in raw_words]


def try_load_seq2seq_model(lang: str, model_root: str, device: str):
    model_path = os.path.join(model_root, f"{lang}_model")
    if not os.path.exists(model_path):
        return None, None

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"[{lang.upper()}] seq2seq load failed -> MFR fallback. Reason: {e}")
        return None, None


def predict_with_seq2seq(
    raw_words: Sequence[str],
    lang: str,
    fmt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_input_length: int = 128,
    max_output_length: int = 64,
    num_beams: int = 2,
) -> List[str]:
    import torch

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
        max_length=max_input_length,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_output_length, num_beams=num_beams)

    preds = [p.strip() for p in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    if len(preds) != len(raw_words):
        # Defensive fallback; the scorer requires token-level length consistency.
        return raw_words
    return preds


def compare_raw_order(reference_rows: Sequence[dict], predictions: Sequence[dict], name: str) -> None:
    ref_raw = [jsonable_list(row.get("raw")) for row in reference_rows]
    pred_raw = [jsonable_list(row.get("raw")) for row in predictions]

    if len(ref_raw) != len(pred_raw):
        raise RuntimeError(
            f"{name} row count mismatch: reference={len(ref_raw)}, predictions={len(pred_raw)}"
        )

    for i, (a, b) in enumerate(zip(ref_raw, pred_raw)):
        if a != b:
            raise RuntimeError(
                f"{name} raw mismatch at row {i}\n"
                f"reference raw: {a}\n"
                f"prediction raw: {b}\n"
                "You are probably predicting a different split than the label file, "
                "or row order was changed."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--valid-path", default=None)
    parser.add_argument("--test-path", default=None, help="Target split to predict. Use validation here for local validation scoring.")
    parser.add_argument("--reference-path", default=None, help="Optional raw-order reference parquet/json. Usually same as --test-path.")
    parser.add_argument("--hf-dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--ko-model-dir", default="artifacts/ko_reranker")
    parser.add_argument("--seq2seq-model-root", default="final_model")
    parser.add_argument("--output-json", default="submission_files/predictions.json")
    parser.add_argument("--zip-path", default="submission.zip")
    parser.add_argument("--use-valid-for-mfr", action="store_true", default=True)
    parser.add_argument("--no-use-valid-for-mfr", action="store_false", dest="use_valid_for_mfr")
    args = parser.parse_args()

    train_rows = load_split(args.train_path, args.hf_dataset, "train")
    valid_rows = load_split(args.valid_path, args.hf_dataset, "test") if args.valid_path else []
    target_rows = load_split(args.test_path, args.hf_dataset, "test")

    mfr_source_rows = list(train_rows)
    if args.use_valid_for_mfr and valid_rows:
        mfr_source_rows.extend(valid_rows)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    ko_bundle = load_bundle(args.ko_model_dir)

    print("============================================================")
    print("All-language prediction with KO reranker")
    print("============================================================")
    print(f"train rows       : {len(train_rows)}")
    print(f"valid rows       : {len(valid_rows)}")
    print(f"target rows      : {len(target_rows)}")
    print(f"use valid for MFR: {args.use_valid_for_mfr and bool(valid_rows)}")
    print(f"device           : {device}")
    print("IMPORTANT: output row order follows --test-path exactly.")
    print("============================================================")

    mfr_cache: Dict[str, Dict[str, str]] = {}
    seq2seq_cache: Dict[str, Tuple[Any, Any]] = {}
    predictions: List[dict] = []

    for row in tqdm(target_rows, desc="[ALL] prediction"):
        lang = row.get("lang")
        raw_for_json = jsonable_list(row.get("raw"))
        raw_words = [str(x) for x in raw_for_json]

        if lang == "ko":
            pred_words = predict_sentence(raw_words, ko_bundle)
        else:
            fmt = STRATEGY.get(lang, "mfr")
            tokenizer, model = (None, None)

            if fmt != "mfr":
                if lang not in seq2seq_cache:
                    seq2seq_cache[lang] = try_load_seq2seq_model(lang, args.seq2seq_model_root, device)
                tokenizer, model = seq2seq_cache[lang]

            if tokenizer is not None and model is not None:
                pred_words = predict_with_seq2seq(raw_words, lang, fmt, tokenizer, model, device)
            else:
                if lang not in mfr_cache:
                    mfr_cache[lang] = build_mfr_dictionary(filter_lang(mfr_source_rows, lang))
                pred_words = predict_with_mfr(raw_words, mfr_cache[lang])

        if len(pred_words) != len(raw_words):
            print(f"[{lang}] length mismatch on row; fallback to raw")
            pred_words = raw_words

        predictions.append({"raw": raw_for_json, "pred": [str(x) for x in pred_words], "lang": lang})

    compare_raw_order(target_rows, predictions, "target-vs-prediction")

    if args.reference_path:
        reference_rows = load_split(args.reference_path, args.hf_dataset, "test")
        compare_raw_order(reference_rows, predictions, "reference-vs-prediction")
        print(f"Reference raw-order check passed: {args.reference_path}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"Saved predictions JSON: {args.output_json}")
    print(f"prediction rows: {len(predictions)}")

    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.output_json, arcname="predictions.json")
        print(f"Saved submission zip: {args.zip_path}")


if __name__ == "__main__":
    main()
