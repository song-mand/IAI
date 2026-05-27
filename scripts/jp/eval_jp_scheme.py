#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the Japanese scheme on validation labels."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import pandas as pd
import torch
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
    evaluate_predictions,
    first_pass_prediction,
    load_json,
    make_byt5_input,
    save_json,
    safety_filter,
)
from sub_jp_scheme import batch_generate


def read_parquet_rows(path: str) -> List[Dict[str, Any]]:
    return pd.read_parquet(path).to_dict("records")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", default=repo_path("train-00000-of-00001.parquet"))
    p.add_argument("--validation_parquet", default=repo_path("validation-00000-of-00001.parquet"))
    p.add_argument("--model_dir", default=repo_path("final_model", "jp_scheme_byt5"))
    p.add_argument("--artifact_path", default=repo_path("final_model", "jp_scheme_artifacts", "jp_scheme_artifacts.json"))
    p.add_argument("--rebuild_artifacts", action="store_true")
    p.add_argument("--no_byt5", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lang_code", default="ja", help="Dataset language label for Japanese. Uploaded parquet uses ja; set jp only if your data uses jp.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_parquet_rows(args.train_parquet)
    valid_rows = read_parquet_rows(args.validation_parquet)

    if args.rebuild_artifacts or not os.path.exists(args.artifact_path):
        artifacts = build_artifacts(train_rows, cfg=ArtifactConfig(), lang=args.lang_code)
        save_json(artifacts, args.artifact_path)
    else:
        artifacts = load_json(args.artifact_path)
        summary = artifacts.get("stats_summary", {})
        if artifacts.get("lang") not in (None, args.lang_code) or summary.get("total_tokens", 0) == 0:
            print("[JP] artifact lang/contents mismatch; rebuilding artifacts")
            artifacts = build_artifacts(train_rows, cfg=ArtifactConfig(), lang=args.lang_code)
            save_json(artifacts, args.artifact_path)
    print(f"[JP] using dataset lang label: {args.lang_code}")
    print("[JP] artifact summary:", artifacts.get("stats_summary"))

    model_bundle = None
    if not args.no_byt5 and os.path.exists(args.model_dir):
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(args.device)
        model.eval()
        model_bundle = (tokenizer, model, args.device)
        print("[JP] ByT5 model loaded")
    else:
        print("[JP] ByT5 disabled or missing. Evaluating copy + high-confidence MFR.")

    pred_rows = []
    byt5_jobs = []
    # First pass all validation rows to allow one global generation batch.
    for row_idx, row in enumerate(valid_rows):
        if row.get("lang") != args.lang_code:
            continue
        raw_words = as_list(row.get("raw"))
        pred_words = list(raw_words)
        for i, raw in enumerate(raw_words):
            first_pred, needs_byt5 = first_pass_prediction(raw, raw_words, i, artifacts)
            pred_words[i] = first_pred
            if needs_byt5 and model_bundle is not None:
                byt5_jobs.append((len(pred_rows), i, make_byt5_input(args.lang_code, raw_words, i), raw_words))
        pred_rows.append({"raw": raw_words, "pred": pred_words, "lang": args.lang_code})

    if byt5_jobs and model_bundle is not None:
        tokenizer, model, device = model_bundle
        texts = [j[2] for j in byt5_jobs]
        outputs = []
        for start in tqdm(range(0, len(texts), args.batch_size), desc="ByT5 eval"):
            outputs.extend(batch_generate(
                tokenizer, model, device,
                texts[start:start + args.batch_size],
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            ))
        for (pred_row_idx, token_i, _, raw_words), out in zip(byt5_jobs, outputs):
            pred_rows[pred_row_idx]["pred"][token_i] = safety_filter(raw_words[token_i], out, raw_words, token_i, artifacts)

    gold_eval_rows = [row for row in valid_rows if row.get("lang") == args.lang_code]
    metrics = evaluate_predictions(gold_eval_rows, pred_rows, lang=args.lang_code)
    print("\n[JA validation]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f} ({v*100:.2f}%)")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
