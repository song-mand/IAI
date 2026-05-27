# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from datasets import load_dataset

from th_utils import ResourceConfig, build_resources, save_pickle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("DATASET", "weerayut/multilexnorm2026-dev-pub"))
    p.add_argument("--split", default=os.environ.get("TRAIN_SPLIT", "train"))
    p.add_argument("--lang", default=os.environ.get("LANG", "th"))
    p.add_argument("--out", default=os.environ.get("TH_RESOURCE_PATH", "models/th/th_resources.pkl"))
    p.add_argument("--summary-out", default=os.environ.get("TH_RESOURCE_SUMMARY", "models/th/th_resources_summary.json"))
    p.add_argument("--min-count", type=int, default=int(os.environ.get("MIN_COUNT", "1")))
    p.add_argument("--top-n", type=int, default=int(os.environ.get("TOP_N_PRINT", "30")))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("=" * 80)
    print("Build Thai resources")
    print("=" * 80)
    print(f"dataset={args.dataset} split={args.split} lang={args.lang}")
    ds = load_dataset(args.dataset, split=args.split)
    rows = [r for r in ds if r.get("lang") == args.lang]
    print(f"rows={len(rows)}")

    res = build_resources(rows, lang=args.lang, config=ResourceConfig(min_count=args.min_count))
    save_pickle(res, args.out)

    changed_pairs = []
    for raw, cand_map in res["error_dict"].items():
        total = sum(cand_map.values())
        best, cnt = max(cand_map.items(), key=lambda x: (x[1], x[0]))
        changed_pairs.append((raw, best, cnt, total, cnt / max(1, total)))
    changed_pairs.sort(key=lambda x: (-x[2], x[0]))

    summary = {
        "lang": args.lang,
        "rows": res["n_rows"],
        "tokens": res["n_tokens"],
        "changed_tokens": res["n_changed"],
        "changed_rate": res["n_changed"] / max(1, res["n_tokens"]),
        "unique_raw": len(res["raw_counter"]),
        "unique_correct": len(res["correct_counter"]),
        "error_dict_keys": len(res["error_dict"]),
        "mfr_changed_keys": sum(1 for r, b in res["mfr"].items() if r != b),
        "top_changed_pairs": [
            {"raw": r, "best": b, "count": c, "total": t, "conf": conf}
            for r, b, c, t, conf in changed_pairs[: args.top_n]
        ],
    }
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved resources: {args.out}")
    print(f"saved summary:   {args.summary_out}")
    print("-" * 80)
    print(f"tokens={summary['tokens']} changed={summary['changed_tokens']} ({summary['changed_rate']*100:.2f}%)")
    print(f"unique_raw={summary['unique_raw']} unique_correct={summary['unique_correct']}")
    print(f"error_dict_keys={summary['error_dict_keys']} mfr_changed_keys={summary['mfr_changed_keys']}")
    print("Top changed pairs:")
    for item in summary["top_changed_pairs"][:10]:
        print(f"  {item['raw']} -> {item['best']}  count={item['count']} total={item['total']} conf={item['conf']:.3f}")


if __name__ == "__main__":
    main()
