# -*- coding: utf-8 -*-
"""Train and validate the Korean contextual MFR reranker.

Example:
  python scripts/ko/train_ko_reranker.py \
    --train-path data/train-00000-of-00001.parquet \
    --valid-path data/validation-00000-of-00001.parquet \
    --model-dir artifacts/ko_reranker \
    --fit-final-with-valid
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, List

from ko_reranker import (
    RerankerConfig,
    evaluate_predictions,
    fit_reranker,
    ko_rows,
    save_bundle,
)


def load_split(path: str | None, hf_dataset: str, split: str) -> List[dict]:
    if path:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")

    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    return [dict(row) for row in ds]


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(f"total          : {int(metrics['total'])}")
    print(f"Baseline LAI   : {metrics['lai'] * 100:.2f}")
    print(f"Accuracy       : {metrics['accuracy'] * 100:.2f}")
    print(f"ERR            : {metrics['err'] * 100:.2f}")
    print(f"gold changed   : {int(metrics['changed_gold'])}")
    print(f"pred changed   : {int(metrics['changed_pred'])}")
    print(f"changed correct: {int(metrics['changed_correct'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--valid-path", default=None)
    parser.add_argument("--hf-dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-dir", default="artifacts/ko_reranker")
    parser.add_argument("--fit-final-with-valid", action="store_true")

    parser.add_argument("--ml-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.25)
    parser.add_argument("--identity-bonus", type=float, default=0.02)
    parser.add_argument("--changed-rate-guard", type=float, default=0.05)
    parser.add_argument("--low-change-keep-raw-bonus", type=float, default=0.12)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    train_all = load_split(args.train_path, args.hf_dataset, "train")
    valid_all = load_split(args.valid_path, args.hf_dataset, "test")

    train_rows = ko_rows(train_all, lang="ko")
    valid_rows = ko_rows(valid_all, lang="ko")
    print(f"ko train rows: {len(train_rows)}")
    print(f"ko valid rows: {len(valid_rows)}")

    config = RerankerConfig(
        ml_weight=args.ml_weight,
        prior_weight=args.prior_weight,
        identity_bonus=args.identity_bonus,
        changed_rate_guard=args.changed_rate_guard,
        low_change_keep_raw_bonus=args.low_change_keep_raw_bonus,
        window=args.window,
    )

    bundle = fit_reranker(train_rows, config)
    if valid_rows:
        metrics = evaluate_predictions(valid_rows, bundle)
        print_metrics("validation", metrics)

        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "valid_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.fit_final_with_valid and valid_rows:
        print("\nRefitting final model with train + validation Korean rows...")
        bundle = fit_reranker(train_rows + valid_rows, config)

    save_bundle(bundle, args.model_dir)
    print(f"\nSaved Korean reranker to: {args.model_dir}/ko_reranker.joblib")


if __name__ == "__main__":
    main()
