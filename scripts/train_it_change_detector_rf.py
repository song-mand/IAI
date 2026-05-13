import argparse
import os
import re
import unicodedata
from collections import Counter, defaultdict

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


LANG = "it"


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_apostrophe(s: str) -> str:
    return s.replace("’", "'").replace("`", "'").replace("´", "'")


def collapse_repeats(s: str, max_repeat: int = 2) -> str:
    return re.sub(
        r"(.)\1{" + str(max_repeat) + r",}",
        lambda m: m.group(1) * max_repeat,
        s,
    )


def make_key(s: str) -> str:
    s = normalize_apostrophe(s)
    s = s.lower()
    s = strip_accents(s)
    s = collapse_repeats(s, max_repeat=2)
    return s


def is_protected_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return True
    if t.startswith("@") or t.startswith("#"):
        return True
    if t.startswith("http://") or t.startswith("https://"):
        return True
    if re.fullmatch(r"\d+([.,:/-]\d+)*", t):
        return True
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True
    return False


def has_long_repetition(token: str) -> bool:
    return re.search(r"(.)\1{2,}", token.lower()) is not None


def has_laughter_pattern(token: str) -> bool:
    t = token.lower()
    return bool(
        re.search(r"(ah){2,}", t)
        or re.search(r"(ha){2,}", t)
        or re.search(r"(haha){1,}", t)
    )


def token_shape(token: str) -> str:
    out = []
    for ch in token:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("0")
        elif ch in "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ":
            out.append("à")
        else:
            out.append(ch)
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(out))


def build_stats(rows):
    raw_counts = defaultdict(Counter)
    key_counts = defaultdict(Counter)
    case_counts = defaultdict(Counter)

    for row in rows:
        if row["lang"] != LANG:
            continue
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            raw_counts[raw][target] += 1
            key_counts[make_key(raw)][target] += 1
            case_counts[target.lower()][target] += 1

    raw_stats = {}
    for raw, counter in raw_counts.items():
        total = sum(counter.values())
        copy = counter.get(raw, 0)
        changed = total - copy
        best_norm, best_count = counter.most_common(1)[0]
        raw_stats[raw] = {
            "total": total,
            "copy": copy,
            "changed": changed,
            "change_prob": changed / total if total else 0.0,
            "copy_prob": copy / total if total else 0.0,
            "best_norm": best_norm,
            "best_count": best_count,
            "best_prob": best_count / total if total else 0.0,
        }

    key_stats = {}
    for key, counter in key_counts.items():
        total = sum(counter.values())
        best_norm, best_count = counter.most_common(1)[0]
        key_stats[key] = {
            "total": total,
            "best_norm": best_norm,
            "best_count": best_count,
            "best_prob": best_count / total if total else 0.0,
        }

    case_stats = {}
    for low, counter in case_counts.items():
        total = sum(counter.values())
        best_norm, best_count = counter.most_common(1)[0]
        case_stats[low] = {
            "total": total,
            "best_norm": best_norm,
            "best_count": best_count,
            "best_prob": best_count / total if total else 0.0,
        }

    return raw_stats, key_stats, case_stats


def get_raw_stat_features(raw: str, raw_stats):
    s = raw_stats.get(raw)
    if s is None:
        return {
            "raw_seen": 0,
            "raw_total": 0,
            "raw_change_prob": 0.0,
            "raw_copy_prob": 0.0,
            "raw_best_is_copy": 0,
            "raw_best_prob": 0.0,
        }

    return {
        "raw_seen": 1,
        "raw_total": min(s["total"], 10),
        "raw_change_prob": s["change_prob"],
        "raw_copy_prob": s["copy_prob"],
        "raw_best_is_copy": int(s["best_norm"] == raw),
        "raw_best_prob": s["best_prob"],
    }


def get_key_stat_features(raw: str, key_stats):
    s = key_stats.get(make_key(raw))
    if s is None:
        return {
            "key_seen": 0,
            "key_total": 0,
            "key_best_prob": 0.0,
            "key_best_is_raw": 0,
        }

    return {
        "key_seen": 1,
        "key_total": min(s["total"], 10),
        "key_best_prob": s["best_prob"],
        "key_best_is_raw": int(s["best_norm"] == raw),
    }


def get_case_stat_features(raw: str, case_stats):
    s = case_stats.get(raw.lower())
    if s is None:
        return {
            "case_seen": 0,
            "case_total": 0,
            "case_best_prob": 0.0,
            "case_best_is_raw": 0,
        }

    return {
        "case_seen": 1,
        "case_total": min(s["total"], 10),
        "case_best_prob": s["best_prob"],
        "case_best_is_raw": int(s["best_norm"] == raw),
    }


def extract_features(raw: str, left: str, right: str, raw_stats, key_stats, case_stats):
    t = raw
    low = t.lower()
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    punct = sum((not ch.isalnum()) for ch in t)

    feats = {
        "bias": 1,
        "raw_lower=" + low: 1,
        "key=" + make_key(t): 1,
        "shape=" + token_shape(t): 1,
        "left_lower=" + left.lower(): 1,
        "right_lower=" + right.lower(): 1,
        "prefix1=" + low[:1]: 1,
        "prefix2=" + low[:2]: 1,
        "prefix3=" + low[:3]: 1,
        "suffix1=" + low[-1:]: 1,
        "suffix2=" + low[-2:]: 1,
        "suffix3=" + low[-3:]: 1,
        "len": min(len(t), 30),
        "letters": min(letters, 30),
        "digits": min(digits, 30),
        "punct": min(punct, 30),
        "is_protected": int(is_protected_token(t)),
        "starts_at": int(t.startswith("@")),
        "starts_hash": int(t.startswith("#")),
        "starts_http": int(t.startswith("http://") or t.startswith("https://")),
        "is_digit_like": int(bool(re.fullmatch(r"\d+([.,:/-]\d+)*", t))),
        "has_long_repetition": int(has_long_repetition(t)),
        "has_laughter_pattern": int(has_laughter_pattern(t)),
        "has_accent": int(strip_accents(t) != t),
        "has_apostrophe": int("'" in normalize_apostrophe(t)),
        "is_all_lower": int(t.islower()),
        "is_all_upper": int(t.isupper()),
        "is_title": int(t.istitle()),
        "has_qkx": int(any(ch in low for ch in ["q", "k", "x"])),
        "has_underscore": int("_" in t),
    }

    feats.update(get_raw_stat_features(raw, raw_stats))
    feats.update(get_key_stat_features(raw, key_stats))
    feats.update(get_case_stat_features(raw, case_stats))
    return feats


def build_examples(rows, raw_stats, key_stats, case_stats):
    X = []
    y = []

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

    return X, np.array(y, dtype=np.int64)


def parse_class_weight(mode: str, change_weight: float):
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode == "balanced_subsample":
        return "balanced_subsample"
    if mode == "custom":
        return {0: 1.0, 1: float(change_weight)}
    raise ValueError(f"unknown class_weight mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./eval_splits_it/it_train.parquet")
    parser.add_argument("--output", type=str, default="./detectors/it_change_detector_rf.joblib")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=14)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--min_samples_split", type=int, default=4)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--class_weight", choices=["none", "custom", "balanced", "balanced_subsample"], default="custom")
    parser.add_argument("--change_weight", type=float, default=3.5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
    raw_stats, key_stats, case_stats = build_stats(train_data)
    X, y = build_examples(train_data, raw_stats, key_stats, case_stats)

    print("examples:", len(y))
    print("COPY examples:", int((y == 0).sum()))
    print("CHANGE examples:", int((y == 1).sum()))
    print("CHANGE rate:", float(y.mean()) if len(y) else 0.0)

    class_weight = parse_class_weight(args.class_weight, args.change_weight)
    print("class_weight:", class_weight)

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features,
            class_weight=class_weight,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )),
    ])

    clf.fit(X, y)

    train_pred = clf.predict(X)
    print("\n[Train detector report: RandomForest / IT]")
    print(classification_report(y, train_pred, target_names=["COPY", "CHANGE"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, train_pred))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    artifact = {
        "model": clf,
        "raw_stats": raw_stats,
        "key_stats": key_stats,
        "case_stats": case_stats,
        "lang": LANG,
        "threshold": 0.5,
        "detector_type": "random_forest",
        "params": vars(args),
    }
    joblib.dump(artifact, args.output)
    print("\nsaved:", args.output)


if __name__ == "__main__":
    main()
