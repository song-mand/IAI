import argparse
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from it_monoise_extend_candidates import (
    LANG,
    feature_dict,
    evaluate_predictions,
    print_metrics,
)


WEAK_SOURCES = {
    "case",
    "key",
    "diacritic",
    "split",
    "split_case",
    "split_diacritic",
}

STRONG_SOURCES = {
    "lookup",
    "mfr",
    "rule",
    "rule_lower",
    "repeat",
    "repeat_key",
    "repeat_case",
    "repeat_diacritic",
    "repeat_diacritic",
    "repeat_exact",
}


def build_mfr_with_conf(rows):
    counts = defaultdict(Counter)

    for row in rows:
        if row["lang"] != LANG:
            continue

        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1

    mfr = {}
    conf = {}
    raw_total = {}

    for raw, counter in counts.items():
        total = sum(counter.values())
        best, best_count = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
        mfr[raw] = best
        conf[raw] = best_count / total if total else 0.0
        raw_total[raw] = total

    return mfr, conf, counts, raw_total


def source_features(prefix, sources):
    feats = {}

    sources = set(sources)

    for s in sources:
        feats[f"{prefix}_source_{s}"] = 1

    feats[f"{prefix}_num_sources"] = len(sources)
    feats[f"{prefix}_weak_only"] = int(len(sources) > 0 and sources.issubset(WEAK_SOURCES))
    feats[f"{prefix}_has_strong"] = int(bool(sources & STRONG_SOURCES))
    feats[f"{prefix}_has_lookup"] = int("lookup" in sources)
    feats[f"{prefix}_has_mfr"] = int("mfr" in sources)
    feats[f"{prefix}_has_rule"] = int("rule" in sources or "rule_lower" in sources)
    feats[f"{prefix}_has_repeat"] = int(any(s.startswith("repeat") for s in sources))
    feats[f"{prefix}_has_case"] = int("case" in sources)
    feats[f"{prefix}_has_key"] = int("key" in sources)
    feats[f"{prefix}_has_diacritic"] = int("diacritic" in sources)

    return feats


def get_candidate_scores(raw_words, artifact, mfr):
    """
    한 문장에 대해:
      - candidate 생성
      - 기존 ranker score 계산
      - token별 group 반환
    """
    gen = artifact["generator"]
    ranker = artifact["ranker"]

    all_features = []
    groups = []

    for i, raw in enumerate(raw_words):
        left = raw_words[i - 1] if i > 0 else "<BOS>"
        right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

        cands = gen.generate(raw)

        base = mfr.get(raw, raw)
        if base not in cands:
            cands[base].add("base_mfr")

        start = len(all_features)
        cand_list = []
        source_list = []

        for cand, sources in cands.items():
            cand_list.append(cand)
            source_list.append(set(sources))
            all_features.append(feature_dict(raw, cand, left, right, sources, gen))

        end = len(all_features)

        groups.append({
            "token_idx": i,
            "raw": raw,
            "left": left,
            "right": right,
            "base": base,
            "start": start,
            "end": end,
            "cand_list": cand_list,
            "source_list": source_list,
        })

    if all_features:
        probs = ranker.predict_proba(all_features)[:, 1]
    else:
        probs = np.array([])

    return groups, probs


def pair_accept_features(
    raw,
    base,
    cand,
    left,
    right,
    base_sources,
    cand_sources,
    gen,
    base_score,
    cand_score,
    best_score,
    is_ranker_best,
    mfr_conf,
    raw_total,
):
    """
    accept classifier용 feature.
    candidate가 base를 override해도 되는지 판단한다.
    """
    feats = {}

    # candidate 자체 feature
    cand_feats = feature_dict(raw, cand, left, right, cand_sources, gen)
    for k, v in cand_feats.items():
        feats[f"cand__{k}"] = v

    # base prediction feature
    base_feats = feature_dict(raw, base, left, right, base_sources, gen)
    for k, v in base_feats.items():
        feats[f"base__{k}"] = v

    # source features
    feats.update(source_features("cand", cand_sources))
    feats.update(source_features("base", base_sources))

    # ranker score relation
    feats["cand_ranker_score"] = cand_score
    feats["base_ranker_score"] = base_score
    feats["score_delta_cand_base"] = cand_score - base_score
    feats["score_delta_cand_best"] = cand_score - best_score
    feats["is_ranker_best"] = int(is_ranker_best)

    # relation between raw / base / candidate
    feats["base_eq_raw"] = int(base == raw)
    feats["cand_eq_raw"] = int(cand == raw)
    feats["cand_eq_base"] = int(cand == base)
    feats["cand_lower_eq_base_lower"] = int(cand.lower() == base.lower())
    feats["cand_lower_eq_raw_lower"] = int(cand.lower() == raw.lower())
    feats["len_diff_cand_base"] = min(abs(len(cand) - len(base)), 40)
    feats["len_diff_cand_raw"] = min(abs(len(cand) - len(raw)), 40)

    # MFR confidence
    feats["mfr_conf_raw"] = mfr_conf.get(raw, 0.0)
    feats["raw_seen_total_log"] = np.log1p(raw_total.get(raw, 0))

    # useful interactions
    feats["weak_only_and_low_mfr_conf"] = int(
        feats.get("cand_weak_only", 0) == 1 and mfr_conf.get(raw, 0.0) < 0.8
    )
    feats["strong_source_and_high_delta"] = int(
        feats.get("cand_has_strong", 0) == 1 and (cand_score - base_score) > 0.05
    )
    feats["case_key_only_risk"] = int(
        cand_sources.issubset({"case", "key", "diacritic"})
    )

    return feats


def collect_train_pairs(rows, base_artifact, mfr, mfr_conf, raw_total, args):
    """
    train token에서 pairwise accept 학습 데이터를 만든다.
    각 candidate에 대해:
      label=1: candidate가 gold이고 base는 gold가 아닐 때
      label=0: 그 외
    """
    gen = base_artifact["generator"]

    X = []
    y = []
    weights = []

    n_accept = 0
    n_reject = 0

    for row in tqdm(rows, desc="build accept train pairs"):
        if row["lang"] != LANG:
            continue

        raw_words = row["raw"]
        gold_words = [
            n if n is not None else r
            for r, n in zip(row["raw"], row["norm"])
        ]

        groups, probs = get_candidate_scores(raw_words, base_artifact, mfr)

        for group in groups:
            i = group["token_idx"]
            raw = group["raw"]
            gold = gold_words[i]
            base = group["base"]
            left = group["left"]
            right = group["right"]

            cand_list = group["cand_list"]
            source_list = group["source_list"]
            start = group["start"]
            end = group["end"]
            local_probs = probs[start:end]

            best_local_idx = int(np.argmax(local_probs))
            best_score = float(local_probs[best_local_idx])

            if base in cand_list:
                base_idx = cand_list.index(base)
                base_score = float(local_probs[base_idx])
                base_sources = source_list[base_idx]
            else:
                base_score = 0.0
                base_sources = {"base_mfr"}

            for j, cand in enumerate(cand_list):
                if cand == base:
                    continue

                cand_score = float(local_probs[j])
                cand_sources = source_list[j]

                label = int(cand == gold and base != gold)

                if label == 1:
                    n_accept += 1
                else:
                    n_reject += 1

                feat = pair_accept_features(
                    raw=raw,
                    base=base,
                    cand=cand,
                    left=left,
                    right=right,
                    base_sources=base_sources,
                    cand_sources=cand_sources,
                    gen=gen,
                    base_score=base_score,
                    cand_score=cand_score,
                    best_score=best_score,
                    is_ranker_best=(j == best_local_idx),
                    mfr_conf=mfr_conf,
                    raw_total=raw_total,
                )

                X.append(feat)
                y.append(label)

                # sample weight
                if raw == gold and cand != gold:
                    # over-change negative: 매우 중요
                    w = args.overchange_negative_weight
                elif label == 1:
                    # useful override positive
                    w = args.positive_weight
                else:
                    w = 1.0

                weights.append(w)

    print("accept train pairs:", len(y))
    print("ACCEPT positives:", n_accept)
    print("REJECT negatives:", n_reject)

    return X, np.array(y, dtype=np.int64), np.array(weights, dtype=np.float32)


def train_accept(args):
    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
    train_data = train_data.filter(lambda x: x["lang"] == LANG)

    print("loading base candidate ranker:", args.base_model)
    base_artifact = joblib.load(args.base_model)

    mfr, mfr_conf, mfr_counts, raw_total = build_mfr_with_conf(train_data)

    X, y, weights = collect_train_pairs(
        rows=train_data,
        base_artifact=base_artifact,
        mfr=mfr,
        mfr_conf=mfr_conf,
        raw_total=raw_total,
        args=args,
    )

    accept_model = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features,
            class_weight=args.class_weight if args.class_weight != "none" else None,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )),
    ])

    accept_model.fit(X, y, clf__sample_weight=weights)

    train_pred = accept_model.predict(X)
    print("\n[Train accept/override classifier]")
    print(classification_report(y, train_pred, target_names=["REJECT", "ACCEPT"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, train_pred))

    artifact = {
        "lang": LANG,
        "base_artifact": base_artifact,
        "accept_model": accept_model,
        "mfr": mfr,
        "mfr_conf": mfr_conf,
        "mfr_counts": mfr_counts,
        "raw_total": raw_total,
        "params": vars(args),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(artifact, args.output)
    print("\nsaved:", args.output)


def predict_sentences(raw_sents, artifact, threshold):
    base_artifact = artifact["base_artifact"]
    accept_model = artifact["accept_model"]
    mfr = artifact["mfr"]
    mfr_conf = artifact["mfr_conf"]
    raw_total = artifact["raw_total"]
    gen = base_artifact["generator"]

    pred_sents = []
    counts = Counter()

    all_pair_features = []
    pair_refs = []
    sentence_states = []

    # 1. ranker score와 pair feature를 전체 batch로 만든다
    for sent_idx, raw_words in enumerate(tqdm(raw_sents, desc="build accept eval pairs")):
        groups, probs = get_candidate_scores(raw_words, base_artifact, mfr)

        pred_words = list(raw_words)
        sentence_states.append({
            "pred_words": pred_words,
            "groups": groups,
            "probs": probs,
        })

        for group in groups:
            raw = group["raw"]
            base = group["base"]
            left = group["left"]
            right = group["right"]
            cand_list = group["cand_list"]
            source_list = group["source_list"]
            start = group["start"]
            end = group["end"]
            local_probs = probs[start:end]

            best_local_idx = int(np.argmax(local_probs))
            best_score = float(local_probs[best_local_idx])

            if base in cand_list:
                base_idx = cand_list.index(base)
                base_score = float(local_probs[base_idx])
                base_sources = source_list[base_idx]
            else:
                base_score = 0.0
                base_sources = {"base_mfr"}

            for j, cand in enumerate(cand_list):
                if cand == base:
                    continue

                cand_score = float(local_probs[j])
                cand_sources = source_list[j]

                feat = pair_accept_features(
                    raw=raw,
                    base=base,
                    cand=cand,
                    left=left,
                    right=right,
                    base_sources=base_sources,
                    cand_sources=cand_sources,
                    gen=gen,
                    base_score=base_score,
                    cand_score=cand_score,
                    best_score=best_score,
                    is_ranker_best=(j == best_local_idx),
                    mfr_conf=mfr_conf,
                    raw_total=raw_total,
                )

                all_pair_features.append(feat)
                pair_refs.append({
                    "sent_idx": sent_idx,
                    "token_idx": group["token_idx"],
                    "raw": raw,
                    "base": base,
                    "cand": cand,
                    "cand_sources": cand_sources,
                })

    # 2. accept prob를 batch로 계산
    if all_pair_features:
        accept_probs = accept_model.predict_proba(all_pair_features)[:, 1]
    else:
        accept_probs = np.array([])

    # 3. token마다 가장 accept 확률이 높은 candidate를 고른다
    best_by_token = {}

    for ref, p in zip(pair_refs, accept_probs):
        key = (ref["sent_idx"], ref["token_idx"])
        if key not in best_by_token or p > best_by_token[key]["p"]:
            best_by_token[key] = {
                "p": float(p),
                "cand": ref["cand"],
                "raw": ref["raw"],
                "base": ref["base"],
                "sources": ref["cand_sources"],
            }

    for sent_idx, state in enumerate(sentence_states):
        pred_words = state["pred_words"]

        for group in state["groups"]:
            token_idx = group["token_idx"]
            raw = group["raw"]
            base = group["base"]
            key = (sent_idx, token_idx)

            if key in best_by_token and best_by_token[key]["p"] >= threshold:
                pred = best_by_token[key]["cand"]
                pred_words[token_idx] = pred
                counts["override_accept"] += 1
                for s in best_by_token[key]["sources"]:
                    counts["accept_source_" + s] += 1
            else:
                pred_words[token_idx] = base
                if base == raw:
                    counts["base_copy"] += 1
                else:
                    counts["base_mfr_change"] += 1

        pred_sents.append(pred_words)

    return pred_sents, counts


def eval_accept(args):
    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
    train_data = train_data.filter(lambda x: x["lang"] == LANG)

    valid_data = load_dataset("parquet", data_files={"valid": args.valid_file})["valid"]
    valid_data = valid_data.filter(lambda x: x["lang"] == LANG)

    raw_sents = []
    gold_sents = []

    for row in valid_data:
        raw_words = row["raw"]
        gold_words = [
            n if n is not None else r
            for r, n in zip(row["raw"], row["norm"])
        ]
        raw_sents.append(raw_words)
        gold_sents.append(gold_words)

    # MFR baseline for comparison
    mfr, _, _, _ = build_mfr_with_conf(train_data)
    mfr_preds = [[mfr.get(w, w) for w in raw_words] for raw_words in raw_sents]
    mfr_metrics = evaluate_predictions(raw_sents, gold_sents, mfr_preds)
    print_metrics("MFR baseline", mfr_metrics, show_errors=False)

    print("loading accept override model:", args.model)
    artifact = joblib.load(args.model)

    pred_sents, counts = predict_sentences(
        raw_sents=raw_sents,
        artifact=artifact,
        threshold=args.accept_threshold,
    )

    print("\nAccept decision counts:")
    for k, v in counts.most_common():
        print(f"  {k:28s} {v}")

    metrics = evaluate_predictions(raw_sents, gold_sents, pred_sents)
    print_metrics("MFR-first + learned accept override", metrics, show_errors=args.verbose)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train_file", type=str, required=True)
    p_train.add_argument("--base_model", type=str, required=True)
    p_train.add_argument("--output", type=str, required=True)

    p_train.add_argument("--n_estimators", type=int, default=500)
    p_train.add_argument("--max_depth", type=int, default=12)
    p_train.add_argument("--min_samples_leaf", type=int, default=2)
    p_train.add_argument("--min_samples_split", type=int, default=4)
    p_train.add_argument("--max_features", type=str, default="sqrt")
    p_train.add_argument(
        "--class_weight",
        choices=["none", "balanced", "balanced_subsample"],
        default="balanced_subsample",
    )
    p_train.add_argument("--positive_weight", type=float, default=3.0)
    p_train.add_argument("--overchange_negative_weight", type=float, default=8.0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--n_jobs", type=int, default=-1)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--train_file", type=str, required=True)
    p_eval.add_argument("--valid_file", type=str, required=True)
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--accept_threshold", type=float, default=0.70)
    p_eval.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        train_accept(args)
    elif args.cmd == "eval":
        eval_accept(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()