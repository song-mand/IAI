#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create submission.zip.

Japanese uses JP-Copy-MFR-ContextByT5. All other languages use plain MFR.
By default, the prediction target is loaded from Hugging Face dataset test split, matching the IT baseline code and avoiding accidental use of a larger local parquet file.
Use --use_local_test only for local experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

def repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)

from jp_scheme_common import (
    ArtifactConfig,
    as_list,
    build_artifacts,
    build_general_mfr,
    first_pass_prediction,
    load_json,
    make_byt5_input,
    save_json,
    safety_filter,
)


def read_parquet_rows(path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict("records")


def read_hf_split_rows(dataset_name: str, split_name: str = "test") -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name)
    return [dict(row) for row in ds[split_name]]


def batch_generate(tokenizer, model, device: str, texts: List[str], batch_size: int, max_new_tokens: int, num_beams: int) -> List[str]:
    outs: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=192).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        outs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outs


def predict_jp_sentence(raw_words: List[str], artifacts: Dict[str, Any], model_bundle=None, batch_size: int = 32, max_new_tokens: int = 64, num_beams: int = 2, lang_code: str = "ja") -> List[str]:
    pred = list(raw_words)
    byt5_jobs: List[Tuple[int, str]] = []

    for i, raw in enumerate(raw_words):
        first_pred, needs_byt5 = first_pass_prediction(raw, raw_words, i, artifacts)
        pred[i] = first_pred
        if needs_byt5 and model_bundle is not None:
            byt5_jobs.append((i, make_byt5_input(lang_code, raw_words, i)))

    if byt5_jobs and model_bundle is not None:
        tokenizer, model, device = model_bundle
        texts = [x[1] for x in byt5_jobs]
        gen = batch_generate(tokenizer, model, device, texts, batch_size, max_new_tokens, num_beams)
        for (i, _), out in zip(byt5_jobs, gen):
            pred[i] = safety_filter(raw_words[i], out, raw_words, i, artifacts)

    return pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", default=repo_path("train-00000-of-00001.parquet"))
    p.add_argument("--validation_parquet", default=repo_path("validation-00000-of-00001.parquet"))
    p.add_argument("--test_parquet", default=repo_path("test-00000-of-00001.parquet"), help="Used only with --use_local_test.")
    p.add_argument("--dataset_name", default="weerayut/multilexnorm2026-dev-pub", help="HF dataset used for the official test split.")
    p.add_argument("--use_local_test", action="store_true", help="Read --test_parquet instead of HF dataset test split. Do not use for Codabench unless the parquet row count/order is known to match.")
    p.add_argument("--model_dir", default=repo_path("final_model", "jp_scheme_byt5"))
    p.add_argument("--artifact_path", default=repo_path("final_model", "jp_scheme_artifacts", "jp_scheme_artifacts.json"))
    p.add_argument("--rebuild_artifacts", action="store_true")
    p.add_argument("--include_validation_for_mfr", action="store_true")
    p.add_argument("--no_byt5", action="store_true", help="Use only copy + high-confidence MFR for Japanese.")
    p.add_argument("--output_dir", default=repo_path("submission_files"))
    p.add_argument("--output_zip", default=repo_path("submission.zip"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--min_mfr_count", type=int, default=3)
    p.add_argument("--min_mfr_best_prob", type=float, default=0.75)
    p.add_argument("--min_mfr_change_rate", type=float, default=0.50)
    p.add_argument("--max_mfr_entropy", type=float, default=1.25)
    p.add_argument("--lang_code", default="ja", help="Dataset language label for Japanese. Uploaded parquet uses ja; set jp only if your data uses jp.")
    p.add_argument(
        "--preserve_row_order",
        action="store_true",
        help="Keep the physical parquet row order. Do NOT use this for Codabench if the official baseline groups predictions by language.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_parquet_rows(args.train_parquet)
    valid_rows = read_parquet_rows(args.validation_parquet) if args.validation_parquet and os.path.exists(args.validation_parquet) else []
    if args.use_local_test:
        test_rows = read_parquet_rows(args.test_parquet)
        print(f"[DATA] using LOCAL test parquet: {args.test_parquet}")
    else:
        test_rows = read_hf_split_rows(args.dataset_name, split_name="test")
        print(f"[DATA] using HF dataset test split: {args.dataset_name}")
    print(f"[DATA] prediction target rows: {len(test_rows):,}")

    artifact_source_rows = list(train_rows)
    if args.include_validation_for_mfr:
        artifact_source_rows.extend(valid_rows)

    cfg = ArtifactConfig(
        min_mfr_count=args.min_mfr_count,
        min_mfr_best_prob=args.min_mfr_best_prob,
        min_mfr_change_rate=args.min_mfr_change_rate,
        max_mfr_entropy=args.max_mfr_entropy,
    )
    if args.rebuild_artifacts or not os.path.exists(args.artifact_path):
        artifacts = build_artifacts(artifact_source_rows, cfg=cfg, lang=args.lang_code)
        save_json(artifacts, args.artifact_path)
        print("[JP] rebuilt artifacts:", args.artifact_path)
    else:
        artifacts = load_json(args.artifact_path)
        summary = artifacts.get("stats_summary", {})
        if artifacts.get("lang") not in (None, args.lang_code) or summary.get("total_tokens", 0) == 0:
            print("[JP] artifact lang/contents mismatch; rebuilding artifacts")
            artifacts = build_artifacts(artifact_source_rows, cfg=cfg, lang=args.lang_code)
            save_json(artifacts, args.artifact_path)
        else:
            print("[JP] loaded artifacts:", args.artifact_path)
    print(f"[JP] using dataset lang label: {args.lang_code}")
    print("[JP] artifact summary:", artifacts.get("stats_summary"))

    model_bundle = None
    if not args.no_byt5 and os.path.exists(args.model_dir):
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(args.device)
        model.eval()
        model_bundle = (tokenizer, model, args.device)
        print("[JP] ByT5 model loaded:", args.model_dir)
    else:
        print("[JP] ByT5 disabled or model not found. Japanese will use copy + high-confidence MFR only.")

    # Build non-JP MFR dictionaries from train or train+validation.
    mfr_source_rows = list(train_rows)
    if args.include_validation_for_mfr:
        mfr_source_rows.extend(valid_rows)

    langs = sorted(set(row.get("lang") for row in test_rows if row.get("lang") != args.lang_code))
    non_jp_mfr = {lang: build_general_mfr(mfr_source_rows, lang=lang) for lang in langs}
    for lang in langs:
        print(f"[{lang.upper()}] MFR entries: {len(non_jp_mfr[lang]):,}")

    def predict_row(row: Dict[str, Any]) -> Dict[str, Any]:
        lang = row.get("lang")
        raw_words = as_list(row.get("raw"))
        if lang == args.lang_code:
            pred_words = predict_jp_sentence(
                raw_words,
                artifacts=artifacts,
                model_bundle=model_bundle,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                lang_code=args.lang_code,
            )
        else:
            mfr = non_jp_mfr.get(lang, {})
            pred_words = [mfr.get(w, w) for w in raw_words]
        return {"raw": raw_words, "pred": pred_words, "lang": lang}

    predictions: List[Dict[str, Any]] = []

    if args.preserve_row_order:
        print("[ORDER] preserving physical parquet row order")
        ordered_rows = test_rows
    else:
        # Match the official baseline submission style: process test rows language by language.
        # This matters because the scorer asserts label['raw'].tolist() == pred['raw'].tolist()
        # before scoring, and Codabench labels are usually ordered like the provided baseline output.
        baseline_lang_order = sorted([
            "en", "da", "de", "es", "hr", "it", "nl", "sl", "sr", "tr",
            "iden", "trde", "id", "ja", "jp", "ko", "th", "vi"
        ])
        present_langs = {row.get("lang") for row in test_rows}
        ordered_langs = [l for l in baseline_lang_order if l in present_langs]
        ordered_langs.extend(sorted(present_langs - set(ordered_langs)))
        print("[ORDER] Codabench/baseline language-grouped order:", ordered_langs)
        ordered_rows = []
        for lang in ordered_langs:
            ordered_rows.extend([row for row in test_rows if row.get("lang") == lang])

    for row in tqdm(ordered_rows, desc="Predict"):
        predictions.append(predict_row(row))

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="predictions.json")
    print("[DONE] wrote", args.output_zip)


if __name__ == "__main__":
    main()
