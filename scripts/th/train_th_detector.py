# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

from datasets import load_dataset

from th_utils import (
    FEATURE_NAMES,
    ResourceConfig,
    build_resources,
    collect_raw_gold,
    evaluate_token_level,
    extract_features,
    rows_to_xy,
    save_pickle,
    seed_everything,
    simple_non_byt5_candidate,
    target_or_raw,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("DATASET", "weerayut/multilexnorm2026-dev-pub"))
    p.add_argument("--split", default=os.environ.get("TRAIN_SPLIT", "train"))
    p.add_argument("--lang", default=os.environ.get("LANG", "th"))
    p.add_argument("--out", default=os.environ.get("TH_DETECTOR_PATH", "models/th/th_detector.joblib"))
    p.add_argument("--resource-out", default=os.environ.get("TH_RESOURCE_PATH", "models/th/th_resources.pkl"))
    p.add_argument("--meta-out", default=os.environ.get("TH_DETECTOR_META", "models/th/th_detector_meta.json"))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    p.add_argument("--valid-ratio", type=float, default=float(os.environ.get("VALID_RATIO", "0.15")))
    p.add_argument("--neg-ratio", type=float, default=float(os.environ.get("NEG_RATIO", "6")), help="Max negative:positive ratio for RF training")
    p.add_argument("--max-train-samples", type=int, default=int(os.environ.get("MAX_TRAIN_SAMPLES", "0")), help="0 = no cap")
    p.add_argument("--n-estimators", type=int, default=int(os.environ.get("RF_N_ESTIMATORS", "700")))
    p.add_argument("--max-depth", default=os.environ.get("RF_MAX_DEPTH", "18"))
    p.add_argument("--min-samples-leaf", type=int, default=int(os.environ.get("RF_MIN_SAMPLES_LEAF", "2")))
    p.add_argument("--max-features", default=os.environ.get("RF_MAX_FEATURES", "sqrt"))
    p.add_argument("--class-weight", default=os.environ.get("RF_CLASS_WEIGHT", "balanced_subsample"))
    p.add_argument("--thresholds", default=os.environ.get("THRESHOLDS", "0.30,0.40,0.50,0.60,0.70,0.80,0.90"))
    p.add_argument("--min-mfr-conf", type=float, default=float(os.environ.get("MIN_MFR_CONF", "0.80")))
    p.add_argument("--min-mfr-count", type=int, default=int(os.environ.get("MIN_MFR_COUNT", "2")))
    p.add_argument("--train-final-on-all", type=int, default=int(os.environ.get("TRAIN_FINAL_ON_ALL", "1")))
    return p.parse_args()


def split_rows(rows, valid_ratio, seed):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_valid = max(1, int(len(rows) * valid_ratio)) if len(rows) > 2 else 1
    return rows[n_valid:], rows[:n_valid]


def downsample_training(X, y, pairs, neg_ratio, max_train_samples, seed):
    pos = [i for i, yy in enumerate(y) if yy == 1]
    neg = [i for i, yy in enumerate(y) if yy == 0]
    rng = random.Random(seed)
    max_neg = int(max(1, len(pos)) * neg_ratio)
    if len(neg) > max_neg:
        neg = rng.sample(neg, max_neg)
    idx = pos + neg
    if max_train_samples and len(idx) > max_train_samples:
        # Keep positives as much as possible, sample the rest.
        if len(pos) >= max_train_samples:
            idx = rng.sample(pos, max_train_samples)
        else:
            rest = [i for i in neg if i not in pos]
            idx = pos + rng.sample(rest, max_train_samples - len(pos))
    rng.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx], [pairs[i] for i in idx]


def make_model(args):
    from sklearn.ensemble import RandomForestClassifier

    max_depth = None if str(args.max_depth).lower() in {"none", "0", ""} else int(args.max_depth)
    max_features = None if str(args.max_features).lower() in {"none", "0", ""} else args.max_features
    class_weight = None if str(args.class_weight).lower() in {"none", "0", ""} else args.class_weight
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=args.seed,
        verbose=0,
    )


def predict_by_threshold(rows, resources, model, threshold, min_mfr_conf, min_mfr_count, lang="th"):
    raw_list, gold_list, pred_list = [], [], []
    prob_values = []
    for row in rows:
        if row.get("lang") != lang:
            continue
        raw_words = [str(x) for x in row.get("raw", [])]
        norm_words = row.get("norm", raw_words)
        sent_len = len(raw_words)
        pred = []
        gold = []
        for i, raw in enumerate(raw_words):
            g = target_or_raw(raw, norm_words[i] if i < len(norm_words) else raw)
            prev_tok = raw_words[i - 1] if i > 0 else ""
            next_tok = raw_words[i + 1] if i + 1 < sent_len else ""
            feat = [extract_features(raw, prev_tok, next_tok, i, sent_len, resources)]
            prob = float(model.predict_proba(feat)[0][1])
            prob_values.append(prob)
            if prob >= threshold:
                cand, _, _ = simple_non_byt5_candidate(raw, resources, min_mfr_conf=min_mfr_conf, min_mfr_count=min_mfr_count)
                pred.append(cand)
            else:
                pred.append(raw)
            gold.append(g)
        raw_list.append(raw_words)
        gold_list.append(gold)
        pred_list.append(pred)
    return raw_list, gold_list, pred_list, prob_values


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("=" * 80)
    print("Train Thai should-change detector")
    print("=" * 80)
    print(vars(args))

    ds = load_dataset(args.dataset, split=args.split)
    rows = [r for r in ds if r.get("lang") == args.lang]
    print(f"loaded rows={len(rows)}")
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)
    print(f"train_rows={len(train_rows)} valid_rows={len(valid_rows)}")

    train_res = build_resources(train_rows, lang=args.lang, config=ResourceConfig())
    X, y, pairs = rows_to_xy(train_rows, train_res, lang=args.lang)
    print(f"raw train tokens={len(y)} positives={sum(y)} ({sum(y)/max(1,len(y))*100:.2f}%)")
    Xs, ys, _ = downsample_training(X, y, pairs, args.neg_ratio, args.max_train_samples, args.seed)
    print(f"sampled train tokens={len(ys)} positives={sum(ys)} negatives={len(ys)-sum(ys)}")

    model = make_model(args)
    model.fit(Xs, ys)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    results = []
    print("-" * 80)
    print("Threshold tuning on validation rows using detector + non-ByT5 candidates")
    print("threshold | ERR | Accuracy | LAI | changed_pred | false_pos")
    for th in thresholds:
        raw_l, gold_l, pred_l, probs = predict_by_threshold(
            valid_rows,
            train_res,
            model,
            th,
            min_mfr_conf=args.min_mfr_conf,
            min_mfr_count=args.min_mfr_count,
            lang=args.lang,
        )
        lai, acc, err = evaluate_token_level(raw_l, gold_l, pred_l, info=False)
        changed_pred = sum(1 for rs, ps in zip(raw_l, pred_l) for r, p in zip(rs, ps) if r != p)
        false_pos = sum(1 for rs, gs, ps in zip(raw_l, gold_l, pred_l) for r, g, p in zip(rs, gs, ps) if r == g and p != r)
        results.append({"threshold": th, "lai": lai, "accuracy": acc, "err": err, "changed_pred": changed_pred, "false_pos": false_pos})
        print(f"{th:8.3f} | {err*100:6.2f} | {acc*100:8.2f} | {lai*100:6.2f} | {changed_pred:12d} | {false_pos:9d}")

    best = max(results, key=lambda d: (d["err"], d["accuracy"], -d["false_pos"]))
    print("-" * 80)
    print(f"best_threshold={best['threshold']:.3f} best_ERR={best['err']*100:.2f}")

    # Train final detector/resources on all Thai training data for submission.
    if args.train_final_on_all:
        full_res = build_resources(rows, lang=args.lang, config=ResourceConfig())
        X_all, y_all, pairs_all = rows_to_xy(rows, full_res, lang=args.lang)
        X_final, y_final, _ = downsample_training(X_all, y_all, pairs_all, args.neg_ratio, args.max_train_samples, args.seed)
        final_model = make_model(args)
        final_model.fit(X_final, y_final)
        resources_to_save = full_res
        model_to_save = final_model
        print(f"final trained on all rows: sampled_tokens={len(y_final)} positives={sum(y_final)}")
    else:
        resources_to_save = train_res
        model_to_save = model

    import joblib
    payload = {
        "model": model_to_save,
        "feature_names": FEATURE_NAMES,
        "threshold": float(best["threshold"]),
        "args": vars(args),
    }
    joblib.dump(payload, args.out)
    save_pickle(resources_to_save, args.resource_out)

    meta = {
        "best": best,
        "results": results,
        "feature_names": FEATURE_NAMES,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "full_rows": len(rows),
        "resource_path": args.resource_out,
        "detector_path": args.out,
    }
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"saved detector:  {args.out}")
    print(f"saved resources: {args.resource_out}")
    print(f"saved meta:      {args.meta_out}")


if __name__ == "__main__":
    main()
