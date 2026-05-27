# -*- coding: utf-8 -*-
"""
Korean contextual candidate reranker for MultiLexNorm-style lexical normalization.

Core idea:
1. Build candidate norms for each raw token from train labels.
2. For each (sentence, target token, candidate norm), train a binary classifier:
      1 if candidate == gold norm else 0
3. At inference time, score every candidate and choose the best one.

This is intentionally conservative: it never generates arbitrary new Korean text.
It only chooses among candidates observed in the labeled training data plus identity(raw).
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import joblib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required. Install it with: pip install scikit-learn joblib"
    ) from e


Row = Dict[str, Any]
CandidateCounts = Dict[str, Dict[str, int]]


@dataclass
class RerankerConfig:
    # Candidate and smoothing settings
    lang: str = "ko"
    alpha: float = 0.5

    # Final score = ml_weight * model_prob + prior_weight * log_prior + bonuses
    ml_weight: float = 1.0
    prior_weight: float = 0.25
    identity_bonus: float = 0.02

    # Conservative guard: if a token was rarely changed in train, require stronger ML prob.
    # Set to 0.0 to disable.
    changed_rate_guard: float = 0.05
    low_change_keep_raw_bonus: float = 0.12

    # Context window for feature construction
    window: int = 3

    # Use all observed labels, including raw==norm, as candidate prior counts.
    # The raw token itself is always added as a candidate even if absent from labels.
    include_identity_candidate: bool = True


@dataclass
class CandidateTables:
    counts: CandidateCounts
    raw_total: Dict[str, int]
    raw_changed_total: Dict[str, int]
    changed_counts: CandidateCounts
    candidates: Dict[str, List[str]]


def _as_python_list(x: Any) -> List[Any]:
    """Convert common array/list-like values to a Python list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    # numpy arrays, pyarrow arrays, pandas Series cells, etc.
    if hasattr(x, "tolist"):
        y = x.tolist()
        return y if isinstance(y, list) else [y]
    return list(x)


def norm_or_raw(raw: str, norm: Any) -> str:
    """Return usable gold norm. Empty/None labels are treated as identity."""
    if norm is None:
        return raw
    if isinstance(norm, float) and math.isnan(norm):
        return raw
    if norm == "":
        return raw
    return str(norm)


def iter_rows(data: Any) -> Iterable[Row]:
    """Yield dict rows from HuggingFace Dataset, pandas DataFrame, or list[dict]."""
    if data is None:
        return
    if hasattr(data, "iterrows"):  # pandas DataFrame
        for _, row in data.iterrows():
            yield row.to_dict()
    else:
        for row in data:
            if hasattr(row, "to_dict"):
                yield row.to_dict()
            else:
                yield dict(row)


def ko_rows(data: Any, lang: str = "ko") -> List[Row]:
    out: List[Row] = []
    for row in iter_rows(data):
        if row.get("lang") != lang:
            continue
        raw = _as_python_list(row.get("raw"))
        norm = _as_python_list(row.get("norm")) if "norm" in row else []
        out.append({"raw": [str(x) for x in raw], "norm": norm, "lang": lang})
    return out


def build_candidate_tables(rows: Sequence[Row], config: RerankerConfig) -> CandidateTables:
    counts: MutableMapping[str, Counter] = defaultdict(Counter)
    changed_counts: MutableMapping[str, Counter] = defaultdict(Counter)
    raw_total: Counter = Counter()
    raw_changed_total: Counter = Counter()

    for row in rows:
        if row.get("lang") != config.lang:
            continue
        raw_words = _as_python_list(row.get("raw"))
        norm_words = _as_python_list(row.get("norm"))
        if not norm_words:
            continue

        for raw, norm in zip(raw_words, norm_words):
            raw = str(raw)
            gold = norm_or_raw(raw, norm)
            counts[raw][gold] += 1
            raw_total[raw] += 1
            if gold != raw:
                changed_counts[raw][gold] += 1
                raw_changed_total[raw] += 1

    candidates: Dict[str, List[str]] = {}
    for raw, counter in counts.items():
        cand_set = set(counter.keys())
        cand_set.update(changed_counts.get(raw, {}).keys())
        if config.include_identity_candidate:
            cand_set.add(raw)
        # Stable order: frequent labels first, then lexical tie-breaker.
        candidates[raw] = sorted(
            cand_set,
            key=lambda c: (-counts[raw].get(c, 0), c != raw, c),
        )

    return CandidateTables(
        counts={k: dict(v) for k, v in counts.items()},
        raw_total=dict(raw_total),
        raw_changed_total=dict(raw_changed_total),
        changed_counts={k: dict(v) for k, v in changed_counts.items()},
        candidates=candidates,
    )


def changed_rate(raw: str, tables: CandidateTables) -> float:
    total = tables.raw_total.get(raw, 0)
    if total <= 0:
        return 0.0
    return tables.raw_changed_total.get(raw, 0) / total


def prior_prob(raw: str, cand: str, tables: CandidateTables, config: RerankerConfig) -> float:
    cands = tables.candidates.get(raw, [raw])
    total = tables.raw_total.get(raw, 0)
    count = tables.counts.get(raw, {}).get(cand, 0)
    denom = total + config.alpha * max(len(cands), 1)
    if denom <= 0:
        return 1.0 if cand == raw else 0.0
    return (count + config.alpha) / denom


def _bin_float(x: float, bins: Sequence[float]) -> str:
    for b in bins:
        if x <= b:
            return f"<=%.3f" % b
    return f">%.3f" % bins[-1]


def make_feature_text(
    raw_words: Sequence[str],
    index: int,
    cand: str,
    tables: CandidateTables,
    config: RerankerConfig,
) -> str:
    raw = str(raw_words[index])
    w = config.window
    left = [str(x) for x in raw_words[max(0, index - w): index]]
    right = [str(x) for x in raw_words[index + 1: index + 1 + w]]
    prev_word = left[-1] if left else "<BOS>"
    next_word = right[0] if right else "<EOS>"

    pp = prior_prob(raw, cand, tables, config)
    cr = changed_rate(raw, tables)
    raw_count = tables.raw_total.get(raw, 0)
    cand_count = tables.counts.get(raw, {}).get(cand, 0)

    # Feature string is used by char/word n-gram TF-IDF.
    # Keep explicit tags because they help the classifier distinguish roles.
    return " ".join([
        f"RAW={raw}",
        f"CAND={cand}",
        f"PREV={prev_word}",
        f"NEXT={next_word}",
        f"LEFT={'|'.join(left) if left else '<BOS>'}",
        f"RIGHT={'|'.join(right) if right else '<EOS>'}",
        f"SENT={' '.join(map(str, raw_words))}",
        f"IS_IDENTITY={int(cand == raw)}",
        f"PRIOR_BIN={_bin_float(pp, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90])}",
        f"CHANGED_RATE_BIN={_bin_float(cr, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75])}",
        f"RAW_COUNT_BIN={min(raw_count, 50)}",
        f"CAND_COUNT_BIN={min(cand_count, 50)}",
        f"RAW_LEN={len(raw)}",
        f"CAND_LEN={len(cand)}",
        f"HAS_JAMO={int(any('ㄱ' <= ch <= 'ㅎ' or 'ㅏ' <= ch <= 'ㅣ' for ch in raw))}",
        f"STARTS_개={int(raw.startswith('개'))}",
        f"ENDS_노={int(raw.endswith('노'))}",
    ])


def make_training_examples(
    rows: Sequence[Row],
    tables: CandidateTables,
    config: RerankerConfig,
) -> Tuple[List[str], List[int]]:
    X: List[str] = []
    y: List[int] = []

    for row in rows:
        if row.get("lang") != config.lang:
            continue
        raw_words = [str(x) for x in _as_python_list(row.get("raw"))]
        norm_words = _as_python_list(row.get("norm"))
        if not norm_words:
            continue

        for i, raw in enumerate(raw_words):
            gold = norm_or_raw(raw, norm_words[i])
            cands = tables.candidates.get(raw, [raw])
            if gold not in cands:
                cands = list(cands) + [gold]

            for cand in cands:
                X.append(make_feature_text(raw_words, i, cand, tables, config))
                y.append(1 if cand == gold else 0)

    return X, y


def build_model() -> Pipeline:
    """Small-data friendly binary candidate scorer."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            ),
        ),
    ])


def fit_reranker(rows: Sequence[Row], config: RerankerConfig | None = None) -> Dict[str, Any]:
    if config is None:
        config = RerankerConfig()
    tables = build_candidate_tables(rows, config)
    X, y = make_training_examples(rows, tables, config)
    if len(set(y)) < 2:
        raise ValueError("Training examples contain only one class. Check Korean labeled rows.")

    model = build_model()
    model.fit(X, y)
    return {"model": model, "tables": tables, "config": config}


def _candidate_ml_probs(model: Pipeline, features: List[str]) -> List[float]:
    if hasattr(model, "predict_proba"):
        return [float(x) for x in model.predict_proba(features)[:, 1]]
    # Fallback, usually unused.
    return [float(x) for x in model.decision_function(features)]


def predict_token(
    raw_words: Sequence[str],
    index: int,
    bundle: Dict[str, Any],
) -> str:
    model: Pipeline = bundle["model"]
    tables: CandidateTables = bundle["tables"]
    config: RerankerConfig = bundle["config"]

    raw = str(raw_words[index])
    cands = list(tables.candidates.get(raw, [raw]))
    if not cands:
        return raw
    if len(cands) == 1:
        return cands[0]

    features = [make_feature_text(raw_words, index, cand, tables, config) for cand in cands]
    ml_probs = _candidate_ml_probs(model, features)

    scores: List[float] = []
    cr = changed_rate(raw, tables)
    for cand, ml_prob in zip(cands, ml_probs):
        pp = prior_prob(raw, cand, tables, config)
        score = config.ml_weight * ml_prob + config.prior_weight * math.log(max(pp, 1e-12))

        if cand == raw:
            score += config.identity_bonus
            if config.changed_rate_guard > 0 and cr < config.changed_rate_guard:
                score += config.low_change_keep_raw_bonus

        scores.append(score)

    best_i = max(range(len(cands)), key=lambda i: scores[i])
    return cands[best_i]


def predict_sentence(raw_words: Sequence[str], bundle: Dict[str, Any]) -> List[str]:
    raw_words = [str(x) for x in raw_words]
    return [predict_token(raw_words, i, bundle) for i in range(len(raw_words))]


def evaluate_predictions(rows: Sequence[Row], bundle: Dict[str, Any]) -> Dict[str, float]:
    total = 0
    correct = 0
    changed = 0
    pred_changed = 0
    changed_correct = 0

    for row in rows:
        if row.get("lang") != bundle["config"].lang:
            continue
        raw_words = [str(x) for x in _as_python_list(row.get("raw"))]
        norm_words = _as_python_list(row.get("norm"))
        if not norm_words:
            continue
        preds = predict_sentence(raw_words, bundle)

        for raw, norm, pred in zip(raw_words, norm_words, preds):
            gold = norm_or_raw(raw, norm)
            total += 1
            if raw != gold:
                changed += 1
            if raw != pred:
                pred_changed += 1
            if pred == gold:
                correct += 1
                if raw != gold:
                    changed_correct += 1

    acc = correct / total if total else 0.0
    lai = (total - changed) / total if total else 0.0
    err = (acc - lai) / (1.0 - lai) if changed else 0.0
    return {
        "total": float(total),
        "accuracy": acc,
        "lai": lai,
        "err": err,
        "changed_gold": float(changed),
        "changed_pred": float(pred_changed),
        "changed_correct": float(changed_correct),
    }


def save_bundle(bundle: Dict[str, Any], model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "ko_reranker.joblib")
    joblib.dump(bundle, path)

    # Human-readable config copy.
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(asdict(bundle["config"]), f, ensure_ascii=False, indent=2)


def load_bundle(model_dir: str) -> Dict[str, Any]:
    path = os.path.join(model_dir, "ko_reranker.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing reranker bundle: {path}")
    return joblib.load(path)


def mfr_predict_sentence(raw_words: Sequence[str], tables: CandidateTables, changed_only: bool = False) -> List[str]:
    preds: List[str] = []
    for raw in raw_words:
        raw = str(raw)
        source = tables.changed_counts if changed_only and tables.changed_counts.get(raw) else tables.counts
        counter = source.get(raw, {})
        if not counter:
            preds.append(raw)
        else:
            preds.append(max(counter.items(), key=lambda x: (x[1], x[0] == raw))[0])
    return preds
