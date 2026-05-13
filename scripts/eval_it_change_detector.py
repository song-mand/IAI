import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_it_change_detector_rf import extract_features, LANG


def build_valid_examples(rows, raw_stats, key_stats, case_stats):
    X = []
    y = []
    tokens = []

    for row in rows:
        if row["lang"] != LANG:
            continue

        raw_words = row["raw"]
        norm_words = row["norm"]

        for i, raw in enumerate(raw_words):
            norm = norm_words[i] if norm_words[i] is not None else raw
            label = int(raw != norm)

            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

            X.append(extract_features(raw, left, right, raw_stats, key_stats, case_stats))
            y.append(label)
            tokens.append((raw, norm, left, right))

    return X, np.array(y, dtype=np.int64), tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_file", type=str, default="./eval_splits_it/it_valid.parquet")
    parser.add_argument("--detector", type=str, default="./detectors/it_change_detector_rf.joblib")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show_errors", action="store_true")
    parser.add_argument("--max_errors", type=int, default=80)
    args = parser.parse_args()

    artifact = joblib.load(args.detector)
    model = artifact["model"]
    raw_stats = artifact["raw_stats"]
    key_stats = artifact["key_stats"]
    case_stats = artifact.get("case_stats", {})

    valid_data = load_dataset("parquet", data_files={"valid": args.valid_file})["valid"]
    X, y, tokens = build_valid_examples(valid_data, raw_stats, key_stats, case_stats)

    prob_change = model.predict_proba(X)[:, 1]
    pred = (prob_change >= args.threshold).astype(np.int64)

    print("examples:", len(y))
    print("true COPY:", int((y == 0).sum()))
    print("true CHANGE:", int((y == 1).sum()))
    print("threshold:", args.threshold)

    print("\n[IT Detector report]")
    print(classification_report(y, pred, target_names=["COPY", "CHANGE"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))

    p, r, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    print("\nSummary:")
    print(f"COPY    precision={p[0]*100:.2f} recall={r[0]*100:.2f} f1={f1[0]*100:.2f} support={support[0]}")
    print(f"CHANGE  precision={p[1]*100:.2f} recall={r[1]*100:.2f} f1={f1[1]*100:.2f} support={support[1]}")

    if args.show_errors:
        print("\n[False positives: predicted CHANGE but gold COPY]")
        shown = 0
        for (raw, norm, left, right), yi, pi, pr in zip(tokens, y, pred, prob_change):
            if yi == 0 and pi == 1:
                print(f"raw={raw} norm={norm} prob_change={pr:.3f} left={left} right={right}")
                shown += 1
                if shown >= args.max_errors:
                    break

        print("\n[False negatives: predicted COPY but gold CHANGE]")
        shown = 0
        for (raw, norm, left, right), yi, pi, pr in zip(tokens, y, pred, prob_change):
            if yi == 1 and pi == 0:
                print(f"raw={raw} norm={norm} prob_change={pr:.3f} left={left} right={right}")
                shown += 1
                if shown >= args.max_errors:
                    break


if __name__ == "__main__":
    main()
