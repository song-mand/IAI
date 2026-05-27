# -*- coding: utf-8 -*-
"""Check whether predictions.json raw rows match a reference parquet/json file.

Use this when scoring.py fails at:
    assert label['raw'].tolist() == pred['raw'].tolist()

Examples:
  python scripts/ko/check_submission_raw_order.py \
    --label-path data/validation-00000-of-00001.parquet \
    --pred-zip submission.zip

  python scripts/ko/check_submission_raw_order.py \
    --label-path data/test-00000-of-00001.parquet \
    --pred-json submission_files/predictions.json
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from typing import Any, List


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


def jsonable_list(x: Any) -> List[Any]:
    out = []
    for v in as_list(x):
        if hasattr(v, "item"):
            try:
                v = v.item()
            except Exception:
                pass
        out.append(v)
    return out


def load_reference(path: str) -> List[List[Any]]:
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return [jsonable_list(x) for x in df["raw"].tolist()]
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [jsonable_list(row["raw"]) for row in data]
    raise ValueError(f"Unsupported reference file: {path}")


def load_predictions(pred_json: str | None, pred_zip: str | None) -> List[List[Any]]:
    if pred_zip:
        with zipfile.ZipFile(pred_zip, "r") as zf:
            names = zf.namelist()
            target = "predictions.json" if "predictions.json" in names else names[0]
            data = json.loads(zf.read(target).decode("utf-8"))
    else:
        assert pred_json is not None
        with open(pred_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    return [jsonable_list(row["raw"]) for row in data]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-path", required=True, help="Reference parquet/json used by scoring.py as label")
    ap.add_argument("--pred-json", default=None)
    ap.add_argument("--pred-zip", default=None)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    if not args.pred_json and not args.pred_zip:
        raise SystemExit("Pass either --pred-json or --pred-zip")

    label_raw = load_reference(args.label_path)
    pred_raw = load_predictions(args.pred_json, args.pred_zip)

    print("============================================================")
    print("Raw order check")
    print("============================================================")
    print(f"label rows: {len(label_raw)}")
    print(f"pred rows : {len(pred_raw)}")

    if len(label_raw) != len(pred_raw):
        print("RESULT: FAIL - row count mismatch")
        print("This usually means you predicted the wrong split.")
        print("Example: validation label + test prediction zip, or vice versa.")
        return

    for i, (a, b) in enumerate(zip(label_raw, pred_raw)):
        if a != b:
            print(f"RESULT: FAIL - first raw mismatch at row {i}")
            print(f"label raw: {a}")
            print(f"pred raw : {b}")
            print("\nLikely causes:")
            print("1) You are scoring validation labels with a test submission zip.")
            print("2) You are using a different test parquet than the label file.")
            print("3) The prediction script grouped/sorted rows instead of preserving input order.")
            return

    print("RESULT: PASS - raw rows match exactly")
    print("If scoring.py still fails after this, inspect how scoring.py loads the files.")


if __name__ == "__main__":
    main()
