import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from it_candidate_ranker import evaluate, make_ranker


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
LANG = "it"

CONFIGS = [
    {
        "name": "R1_mfr_only",
        "use_casefold": False,
        "use_diacritics": False,
        "use_repeat": False,
        "use_abbrev": False,
        "C": 1.0,
    },
    {
        "name": "R2_mfr_casefold",
        "use_casefold": True,
        "use_diacritics": False,
        "use_repeat": False,
        "use_abbrev": False,
        "C": 1.0,
    },
    {
        "name": "R3_mfr_casefold_diacritics",
        "use_casefold": True,
        "use_diacritics": True,
        "use_repeat": False,
        "use_abbrev": False,
        "C": 1.0,
    },
    {
        "name": "R4_plus_repeat",
        "use_casefold": True,
        "use_diacritics": True,
        "use_repeat": True,
        "use_abbrev": False,
        "C": 1.0,
    },
    {
        "name": "R5_plus_abbrev",
        "use_casefold": True,
        "use_diacritics": True,
        "use_repeat": True,
        "use_abbrev": True,
        "C": 1.0,
    },
]


def get_lang_rows(split, lang: str) -> List[Dict[str, Any]]:
    rows = []

    for row in split:
        if row["lang"] != lang:
            continue

        if not isinstance(row.get("raw"), list):
            continue
        if not isinstance(row.get("norm"), list):
            continue
        if len(row["raw"]) != len(row["norm"]):
            continue

        rows.append(row)

    return rows


def kfold_split(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    folds = []
    for fold_idx in range(k):
        dev = [r for i, r in enumerate(rows) if i % k == fold_idx]
        train = [r for i, r in enumerate(rows) if i % k != fold_idx]
        folds.append((train, dev))

    return folds


def rows_to_sents(rows: List[Dict[str, Any]]):
    raw = [r["raw"] for r in rows]
    gold = [[rr if nn is None else nn for rr, nn in zip(r["raw"], r["norm"])] for r in rows]
    return raw, gold


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--lang", default=LANG)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="scripts/it/ranker_artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 50)
    print("1. Load dataset")
    print("=" * 50)

    ds = load_dataset(args.dataset)
    rows = get_lang_rows(ds["train"], args.lang)

    print(f"IT train rows: {len(rows)}")
    print(f"Total tokens: {sum(len(r['raw']) for r in rows)}")

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    folds = kfold_split(rows, args.kfold, args.seed)

    all_results = []
    best = None

    print("=" * 50)
    print("2. Cross validation")
    print("=" * 50)

    for config in CONFIGS:
        for threshold in thresholds:
            fold_metrics = []

            print(f"\nConfig={config['name']} threshold={threshold}")

            for fold_idx, (train_rows, dev_rows) in enumerate(tqdm(folds, desc="folds")):
                ranker = make_ranker(
                    train_rows=train_rows,
                    config=config,
                    threshold=threshold,
                )

                pred = ranker.predict_rows(dev_rows)
                raw_sents, gold_sents = rows_to_sents(dev_rows)
                metrics = evaluate(raw_sents, gold_sents, pred)
                fold_metrics.append(metrics)

            avg_lai = sum(m["lai"] for m in fold_metrics) / len(fold_metrics)
            avg_acc = sum(m["accuracy"] for m in fold_metrics) / len(fold_metrics)
            avg_err = sum(m["err"] for m in fold_metrics) / len(fold_metrics)

            result = {
                "config": config,
                "threshold": threshold,
                "lai": avg_lai,
                "accuracy": avg_acc,
                "err": avg_err,
                "fold_metrics": fold_metrics,
            }
            all_results.append(result)

            print(
                f"  AVG LAI={avg_lai * 100:.2f} "
                f"ACC={avg_acc * 100:.2f} "
                f"ERR={avg_err * 100:.2f}"
            )

            if best is None or avg_err > best["err"]:
                best = result

    print("=" * 50)
    print("3. Best CV result")
    print("=" * 50)
    print(f"config:    {best['config']['name']}")
    print(f"threshold: {best['threshold']}")
    print(f"LAI:       {best['lai'] * 100:.2f}")
    print(f"ACC:       {best['accuracy'] * 100:.2f}")
    print(f"ERR:       {best['err'] * 100:.2f}")

    print("=" * 50)
    print("4. Train final ranker on all IT train")
    print("=" * 50)

    final_ranker = make_ranker(
        train_rows=rows,
        config=best["config"],
        threshold=best["threshold"],
    )

    model_path = os.path.join(args.out_dir, "it_ranker.pkl")
    result_path = os.path.join(args.out_dir, "cv_results.json")
    best_path = os.path.join(args.out_dir, "best_config.json")

    final_ranker.save(model_path)

    save_json(result_path, all_results)
    save_json(best_path, {
        "best_config": best["config"],
        "threshold": best["threshold"],
        "lai": best["lai"],
        "accuracy": best["accuracy"],
        "err": best["err"],
    })

    print(f"Saved model:      {model_path}")
    print(f"Saved CV results: {result_path}")
    print(f"Saved best config:{best_path}")


if __name__ == "__main__":
    main()