# -*- coding: utf-8 -*-
"""Create Korean predictions with the contextual MFR reranker.

This script outputs only Korean rows. It is useful for testing the Korean module
or merging Korean predictions into a larger all-language submission pipeline.

Example:
  python scripts/ko/sub_ko_reranker.py \
    --test-path data/test-00000-of-00001.parquet \
    --model-dir artifacts/ko_reranker \
    --output-json submission_files/ko_predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from typing import Any, List

from tqdm import tqdm

from ko_reranker import ko_rows, load_bundle, predict_sentence


def load_split(path: str | None, hf_dataset: str, split: str) -> List[dict]:
    if path:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")

    from datasets import load_dataset
    ds = load_dataset(hf_dataset, split=split)
    return [dict(row) for row in ds]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--hf-dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-dir", default="artifacts/ko_reranker")
    parser.add_argument("--output-json", default="submission_files/ko_predictions.json")
    parser.add_argument("--zip-path", default=None)
    args = parser.parse_args()

    bundle = load_bundle(args.model_dir)
    test_all = load_split(args.test_path, args.hf_dataset, "test")
    test_rows = ko_rows(test_all, lang="ko")

    print(f"ko test rows: {len(test_rows)}")
    predictions = []
    for row in tqdm(test_rows, desc="[KO] reranker prediction"):
        raw_words = [str(x) for x in row["raw"]]
        pred_words = predict_sentence(raw_words, bundle)
        predictions.append({"raw": raw_words, "pred": pred_words, "lang": "ko"})

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    print(f"Saved Korean predictions: {args.output_json}")

    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.output_json, arcname="predictions.json")
        print(f"Saved zip: {args.zip_path}")


if __name__ == "__main__":
    main()
