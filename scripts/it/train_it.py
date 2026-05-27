import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

from it_normalizer import (
    DEFAULT_CONFIG,
    ITConservativeNormalizer,
    build_artifacts_from_rows,
    evaluate_metrics,
    merge_config,
)


ABLATION_CONFIGS = [
    (
        "A1_exact_mfr",
        {
            "use_exact_mfr": True,
            "use_casefold_mfr": False,
            "use_diacritics": False,
            "use_safe_abbrev": False,
            "use_context_abbrev": False,
            "use_repeat": False,
            "use_capitalization": False,
            "use_split": False,
            "decision_margin": 1.15,
        },
    ),
    (
        "A3_exact_casefold",
        {
            "use_exact_mfr": True,
            "use_casefold_mfr": True,
            "use_diacritics": False,
            "use_safe_abbrev": False,
            "use_context_abbrev": False,
            "use_repeat": False,
            "use_capitalization": False,
            "use_split": False,
            "decision_margin": 1.20,
        },
    ),
    (
        "A4_plus_diacritics",
        {
            "use_exact_mfr": True,
            "use_casefold_mfr": True,
            "use_diacritics": True,
            "use_safe_abbrev": False,
            "use_context_abbrev": False,
            "use_repeat": False,
            "use_capitalization": False,
            "use_split": False,
            "decision_margin": 1.20,
        },
    ),
    (
        "A5_plus_safe_abbrev",
        {
            "use_exact_mfr": True,
            "use_casefold_mfr": True,
            "use_diacritics": True,
            "use_safe_abbrev": True,
            "use_context_abbrev": False,
            "use_repeat": False,
            "use_capitalization": False,
            "use_split": False,
            "decision_margin": 1.25,
        },
    ),
    (
        "A6_plus_repeat",
        {
            "use_exact_mfr": True,
            "use_casefold_mfr": True,
            "use_diacritics": True,
            "use_safe_abbrev": True,
            "use_context_abbrev": False,
            "use_repeat": True,
            "use_capitalization": False,
            "use_split": False,
            "decision_margin": 1.30,
        },
    ),
    (
        "A7_plus_context",
        {
            "use_exact_mfr": True,
            "use_casefold_mfr": True,
            "use_diacritics": True,
            "use_safe_abbrev": True,
            "use_context_abbrev": True,
            "use_repeat": True,
            "use_capitalization": False,
            "use_split": False,
            "decision_margin": 1.35,
        },
    ),
    (
        "A8_plus_capitalization",
        {
            "use_exact_mfr": True,
            "use_casefold_mfr": True,
            "use_diacritics": True,
            "use_safe_abbrev": True,
            "use_context_abbrev": True,
            "use_repeat": True,
            "use_capitalization": True,
            "use_split": False,
            "decision_margin": 1.45,
        },
    ),
]


def rows_for_lang(split, lang: str) -> List[Dict[str, Any]]:
    rows = []

    for row in split:
        if row.get("lang") != lang:
            continue

        raw = row.get("raw")
        norm = row.get("norm")

        if not isinstance(raw, list):
            continue
        if not isinstance(norm, list):
            continue
        if len(raw) == 0:
            continue
        if len(raw) != len(norm):
            continue

        rows.append(row)

    return rows


def split_train_dev(
    rows: List[Dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    dev_size = max(1, int(len(rows) * dev_ratio))
    dev_rows = rows[:dev_size]
    train_rows = rows[dev_size:]

    return train_rows, dev_rows


def predict_rows(normalizer: ITConservativeNormalizer, rows: List[Dict[str, Any]]) -> List[List[str]]:
    return [normalizer.normalize_sentence(row["raw"]) for row in rows]


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--lang", default="it")
    parser.add_argument("--out-dir", default="scripts/it/artifacts")
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 40)
    print("Load dataset")
    print("=" * 40)

    dataset = load_dataset(args.dataset)
    all_train_rows = rows_for_lang(dataset[args.train_split], args.lang)

    print(f"usable train rows: {len(all_train_rows)}")

    if not all_train_rows:
        raise RuntimeError(f"No usable rows for lang={args.lang}")

    train_rows, dev_rows = split_train_dev(
        all_train_rows,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    print(f"inner train rows: {len(train_rows)}")
    print(f"inner dev rows:   {len(dev_rows)}")

    raw_sents = [row["raw"] for row in dev_rows]
    gold_sents = [row["norm"] for row in dev_rows]

    results = []
    best = None

    print("=" * 40)
    print("Ablation on holdout dev")
    print("=" * 40)

    for name, override in ABLATION_CONFIGS:
        artifacts = build_artifacts_from_rows(train_rows, prefer_identity=True)
        config = merge_config(DEFAULT_CONFIG, override)
        artifacts["config"] = config

        normalizer = ITConservativeNormalizer(artifacts)
        pred_sents = predict_rows(normalizer, dev_rows)
        metrics = evaluate_metrics(raw_sents, gold_sents, pred_sents)

        row = {
            "name": name,
            "lai": metrics["lai"],
            "accuracy": metrics["accuracy"],
            "err": metrics["err"],
            "total": metrics["total"],
            "changed": metrics["changed"],
            "config": config,
        }

        results.append(row)

        print(
            f"{name:24s} "
            f"LAI={metrics['lai'] * 100:6.2f} "
            f"ACC={metrics['accuracy'] * 100:6.2f} "
            f"ERR={metrics['err'] * 100:6.2f} "
            f"changed={metrics['changed']}"
        )

        if best is None or metrics["err"] > best["err"]:
            best = row

    print("=" * 40)
    print("Best config")
    print("=" * 40)
    print(
        f"{best['name']} | "
        f"ACC={best['accuracy'] * 100:.2f} | "
        f"ERR={best['err'] * 100:.2f}"
    )

    print("=" * 40)
    print("Build final artifact using all train rows")
    print("=" * 40)

    final_artifacts = build_artifacts_from_rows(all_train_rows, prefer_identity=True)
    final_artifacts["config"] = best["config"]
    final_artifacts["best_ablation"] = best["name"]
    final_artifacts["best_metrics_on_holdout"] = {
        "lai": best["lai"],
        "accuracy": best["accuracy"],
        "err": best["err"],
        "changed": best["changed"],
        "total": best["total"],
    }

    final_normalizer = ITConservativeNormalizer(final_artifacts)
    final_preds = predict_rows(final_normalizer, dev_rows)
    final_metrics = evaluate_metrics(raw_sents, gold_sents, final_preds)

    print(
        "Final artifact sanity on same holdout: "
        f"LAI={final_metrics['lai'] * 100:.2f} "
        f"ACC={final_metrics['accuracy'] * 100:.2f} "
        f"ERR={final_metrics['err'] * 100:.2f}"
    )

    artifact_path = os.path.join(args.out_dir, "it_norm_artifacts.json")
    results_path = os.path.join(args.out_dir, "it_ablation_results.json")
    pred_path = os.path.join(args.out_dir, "it_dev_predictions.json")

    save_json(artifact_path, final_artifacts)
    save_json(results_path, results)

    prediction_dump = []

    for row, pred in zip(dev_rows, final_preds):
        prediction_dump.append(
            {
                "raw": row["raw"],
                "gold": row["norm"],
                "pred": pred,
                "lang": row["lang"],
            }
        )

    save_json(pred_path, prediction_dump)

    print("=" * 40)
    print("Saved")
    print("=" * 40)
    print(artifact_path)
    print(results_path)
    print(pred_path)


if __name__ == "__main__":
    main()