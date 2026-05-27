#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/es/train_es_oof_ranker.py

Spanish OOF-ranker training code.

This script trains/saves:
  - ./detectors/es_resources_oof.joblib
  - ./detectors/es_change_detector_rf.joblib
  - ./detectors/es_candidate_ranker_oof_rf.joblib

It does NOT retrain ByT5. It assumes your existing model remains at:
  - ./final_model/es_model

Required existing file in the same folder:
  - scripts/es/es_rules.py
"""

import argparse
import os
import random
import sys
from collections import Counter

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from es_rules import (  # noqa: E402
    build_stats,
    candidate_features,
    detector_features,
    generate_candidates,
    target_of,
)

LANG = "es"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_rows(train_file: str, dataset_name: str):
    if train_file and os.path.exists(train_file):
        print(f"Loading local train file: {train_file}")
        return load_dataset("parquet", data_files={"train": train_file})["train"]
    print(f"Loading HF dataset: {dataset_name}")
    return load_dataset(dataset_name, split="train")


def get_lang_rows(rows, lang=LANG):
    return [row for row in rows if row["lang"] == lang]


def make_folds(n: int, k: int, seed: int):
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    folds = [[] for _ in range(k)]
    for i, idx in enumerate(idxs):
        folds[i % k].append(idx)
    return folds


def build_detector_examples(rows, resources):
    X, y = [], []
    for row in rows:
        raw_words = list(row["raw"])
        norm_words = list(row["norm"])
        for i, raw in enumerate(raw_words):
            gold = target_of(raw, norm_words[i])
            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
            X.append(detector_features(raw, left, right, resources))
            y.append(int(raw != gold))
    return X, np.array(y, dtype=np.int64)


def train_detector(rows, resources, args):
    X, y = build_detector_examples(rows, resources)

    print("\n[ES RF change detector]")
    print("examples:", len(y))
    print("COPY:", int((y == 0).sum()))
    print("CHANGE:", int((y == 1).sum()))
    print("CHANGE rate:", float(y.mean()) if len(y) else 0.0)

    if args.detector_class_weight == "none":
        class_weight = None
    elif args.detector_class_weight == "custom":
        class_weight = {0: 1.0, 1: args.detector_change_weight}
    else:
        class_weight = args.detector_class_weight

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=args.detector_n_estimators,
            max_depth=args.detector_max_depth if args.detector_max_depth > 0 else None,
            min_samples_leaf=args.detector_min_samples_leaf,
            min_samples_split=args.detector_min_samples_split,
            max_features=args.detector_max_features,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )),
    ])

    clf.fit(X, y)
    pred = clf.predict(X)

    print("\n[Train detector report]")
    print(classification_report(y, pred, target_names=["COPY", "CHANGE"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))

    os.makedirs(os.path.dirname(args.output_detector), exist_ok=True)
    artifact = {
        "model": clf,
        "threshold": args.detector_threshold,
        "lang": LANG,
        "params": vars(args),
    }
    joblib.dump(artifact, args.output_detector)
    print(f"Saved detector: {args.output_detector}")


def generate_candidates_wrapper(raw, resources):
    return generate_candidates(
        raw,
        mfr=resources["mfr"],
        mfr_conf=resources["mfr_conf"],
        key_map=resources["key_map"],
    )


def candidate_upperbound(rows, resources):
    total = hit = 0
    changed_total = changed_hit = 0
    source_hits = Counter()

    for row in rows:
        raw_words = list(row["raw"])
        norm_words = list(row["norm"])
        for raw, norm in zip(raw_words, norm_words):
            gold = target_of(raw, norm)
            cands = generate_candidates_wrapper(raw, resources)
            cand_values = [c for _s, c in cands]
            total += 1
            if raw != gold:
                changed_total += 1
            if gold in cand_values:
                hit += 1
                if raw != gold:
                    changed_hit += 1
                for source, cand in cands:
                    if cand == gold:
                        source_hits[source] += 1
                        break

    return total, hit, changed_total, changed_hit, source_hits


def build_ranker_examples_from_rows(rows, resources, args, fold_name):
    X, y, w = [], [], []
    rng = random.Random(args.seed + abs(hash(fold_name)) % 100000)
    counts = Counter()
    source_hits = Counter()

    for row in rows:
        raw_words = list(row["raw"])
        norm_words = list(row["norm"])

        for i, raw in enumerate(raw_words):
            gold = target_of(raw, norm_words[i])
            changed = raw != gold
            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

            cands = generate_candidates_wrapper(raw, resources)
            cand_values = [c for _s, c in cands]
            gold_in_candidates = gold in cand_values
            noncopy_exists = any(c != raw for _s, c in cands)

            if gold_in_candidates:
                for source, cand in cands:
                    if cand == gold:
                        source_hits[source] += 1
                        break

            if changed:
                use_token = True
                counts["changed_token"] += 1
                if not gold_in_candidates:
                    counts["changed_gold_missing"] += 1
                    if args.ranker_gold_force:
                        cands.append(("gold_forcing", gold))
            else:
                # Keep hard unchanged tokens and downsample easy copy positives.
                hard_unchanged = noncopy_exists
                use_token = hard_unchanged or (rng.random() < args.ranker_unchanged_keep_prob)
                if hard_unchanged:
                    counts["unchanged_hard"] += 1
                else:
                    counts["unchanged_easy"] += 1

            if not use_token:
                continue

            for source, cand in cands:
                label = int(cand == gold)
                X.append(candidate_features(raw, cand, source, left, right, resources))
                y.append(label)

                if changed:
                    if label:
                        weight = args.changed_positive_weight
                    elif cand == raw:
                        weight = args.changed_copy_wrong_weight
                    else:
                        weight = args.changed_wrong_weight
                else:
                    if label:
                        weight = args.unchanged_copy_positive_weight
                    else:
                        weight = args.overchange_negative_weight
                w.append(weight)

    print(f"\n[{fold_name}] ranker example stats")
    for k, v in counts.most_common():
        print(f"  {k:28s} {v}")
    print(f"[{fold_name}] gold source hits")
    for k, v in source_hits.most_common():
        print(f"  {k:28s} {v}")

    return X, y, w


def build_ranker_examples_oof(rows, args):
    folds = make_folds(len(rows), args.ranker_oof_folds, args.seed)

    all_X, all_y, all_w = [], [], []
    total = hit = 0
    changed_total = changed_hit = 0
    source_hits = Counter()

    for fold_id, heldout_idxs in enumerate(folds):
        heldout_set = set(heldout_idxs)
        train_rows = [rows[i] for i in range(len(rows)) if i not in heldout_set]
        heldout_rows = [rows[i] for i in heldout_idxs]

        fold_resources = build_stats(train_rows, LANG)

        t, h, ct, ch, sh = candidate_upperbound(heldout_rows, fold_resources)
        total += t
        hit += h
        changed_total += ct
        changed_hit += ch
        source_hits.update(sh)

        X, y, w = build_ranker_examples_from_rows(
            heldout_rows,
            fold_resources,
            args,
            fold_name=f"OOF-fold-{fold_id}",
        )
        all_X.extend(X)
        all_y.extend(y)
        all_w.extend(w)

    print("\n[OOF candidate upperbound before gold forcing]")
    print(f"candidate upperbound: {hit / total * 100 if total else 0:.2f}%")
    print(f"changed candidate upperbound: {changed_hit / changed_total * 100 if changed_total else 0:.2f}%")
    print("source gold hits:")
    for k, v in source_hits.most_common():
        print(f"  {k:28s} {v}")

    return np.array(all_X, dtype=object).tolist(), np.array(all_y, dtype=np.int64), np.array(all_w, dtype=np.float32)


def train_oof_ranker(rows, args):
    X, y, sample_weight = build_ranker_examples_oof(rows, args)

    print("\n[ES OOF RF candidate ranker]")
    print("examples:", len(y))
    print("positive:", int((y == 1).sum()))
    print("negative:", int((y == 0).sum()))
    print("sample_weight sum:", float(sample_weight.sum()))

    if args.ranker_class_weight == "none":
        class_weight = None
    elif args.ranker_class_weight == "custom":
        class_weight = {0: 1.0, 1: args.ranker_positive_weight}
    else:
        class_weight = args.ranker_class_weight

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=args.ranker_n_estimators,
            max_depth=args.ranker_max_depth if args.ranker_max_depth > 0 else None,
            min_samples_leaf=args.ranker_min_samples_leaf,
            min_samples_split=args.ranker_min_samples_split,
            max_features=args.ranker_max_features,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )),
    ])

    clf.fit(X, y, clf__sample_weight=sample_weight)
    pred = clf.predict(X)

    print("\n[Train OOF ranker report]")
    print(classification_report(y, pred, target_names=["WRONG", "GOLD"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))

    os.makedirs(os.path.dirname(args.output_ranker), exist_ok=True)
    artifact = {
        "model": clf,
        "candidate_margin": args.candidate_margin,
        "ranker_threshold": args.ranker_threshold,
        "lang": LANG,
        "training": "oof",
        "params": vars(args),
    }
    joblib.dump(artifact, args.output_ranker)
    print(f"Saved OOF ranker: {args.output_ranker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./data/train-00000-of-00001.parquet")
    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")

    parser.add_argument("--output_resources", type=str, default="./detectors/es_resources_oof.joblib")
    parser.add_argument("--output_detector", type=str, default="./detectors/es_change_detector_rf.joblib")
    parser.add_argument("--output_ranker", type=str, default="./detectors/es_candidate_ranker_oof_rf.joblib")

    parser.add_argument("--seed", type=int, default=5)

    # Detector params
    parser.add_argument("--detector_n_estimators", type=int, default=600)
    parser.add_argument("--detector_max_depth", type=int, default=12)
    parser.add_argument("--detector_min_samples_leaf", type=int, default=1)
    parser.add_argument("--detector_min_samples_split", type=int, default=4)
    parser.add_argument("--detector_max_features", type=str, default="sqrt")
    parser.add_argument(
        "--detector_class_weight",
        choices=["none", "custom", "balanced", "balanced_subsample"],
        default="custom",
    )
    parser.add_argument("--detector_change_weight", type=float, default=3.5)
    parser.add_argument("--detector_threshold", type=float, default=0.43)

    # OOF ranker params
    parser.add_argument("--ranker_oof_folds", type=int, default=5)
    parser.add_argument("--ranker_unchanged_keep_prob", type=float, default=0.20)
    parser.add_argument("--ranker_gold_force", action="store_true")

    parser.add_argument("--ranker_n_estimators", type=int, default=800)
    parser.add_argument("--ranker_max_depth", type=int, default=14)
    parser.add_argument("--ranker_min_samples_leaf", type=int, default=1)
    parser.add_argument("--ranker_min_samples_split", type=int, default=4)
    parser.add_argument("--ranker_max_features", type=str, default="sqrt")
    parser.add_argument(
        "--ranker_class_weight",
        choices=["none", "custom", "balanced", "balanced_subsample"],
        default="none",
    )
    parser.add_argument("--ranker_positive_weight", type=float, default=1.0)
    parser.add_argument("--ranker_threshold", type=float, default=0.25)
    parser.add_argument("--candidate_margin", type=float, default=0.05)

    # Sample weights for ranker
    parser.add_argument("--changed_positive_weight", type=float, default=4.0)
    parser.add_argument("--changed_copy_wrong_weight", type=float, default=3.0)
    parser.add_argument("--changed_wrong_weight", type=float, default=1.5)
    parser.add_argument("--unchanged_copy_positive_weight", type=float, default=0.3)
    parser.add_argument("--overchange_negative_weight", type=float, default=2.0)

    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()
    set_seed(args.seed)

    rows = load_rows(args.train_file, args.dataset_name)
    es_rows = get_lang_rows(rows, LANG)
    print(f"[ES] rows: {len(es_rows)}")

    final_resources = build_stats(es_rows, LANG)
    os.makedirs(os.path.dirname(args.output_resources), exist_ok=True)
    joblib.dump(final_resources, args.output_resources)
    print(f"Saved final ES resources: {args.output_resources}")

    train_detector(es_rows, final_resources, args)
    train_oof_ranker(es_rows, args)


if __name__ == "__main__":
    main()
