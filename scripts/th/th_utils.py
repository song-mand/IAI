# -*- coding: utf-8 -*-
"""Thai-specific utilities for MultiLexNorm lexical normalization.

This module intentionally keeps the Thai system conservative:
- build extendable MFR / error dictionaries from the training split
- normalize obvious Thai Unicode / mark-order issues as candidate generation
- extract features for a token-level should-change classifier
- provide simple scoring helpers used by training and submission scripts
"""
from __future__ import annotations

import math
import os
import pickle
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"[0-9]")
URL_RE = re.compile(r"^(https?://|www\.|[A-Za-z0-9_.-]+\.[A-Za-z]{2,}/?)")
SPACE_RE = re.compile(r"\s")

# Thai character classes used by the normalization routine.
LEADING_VOWELS = "เแโใไ"  # L
TONE_MARKS = "่้๊๋"        # T
HANGING_VOWELS = "ัิีึืุู็ํ"  # H, include nikkhahit
FOLLOWING_VOWELS = "ะาำๅ"      # F
THAI_MARKS = TONE_MARKS + HANGING_VOWELS
ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"

# A broad set of Thai base chars. This intentionally includes consonants and
# a few standalone Thai symbols, but excludes common vowel/tone combining marks.
BASE_THAI_CHARS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฯๆ"

FEATURE_NAMES = [
    "len",
    "log_freq",
    "thai_ratio",
    "latin_ratio",
    "digit_ratio",
    "has_thai",
    "has_latin",
    "has_digit",
    "is_url_like",
    "is_hashtag",
    "is_mention",
    "is_all_thai",
    "is_mixed_script",
    "num_tone",
    "num_hanging_vowel",
    "num_leading_vowel",
    "num_following_vowel",
    "has_repeated_char",
    "has_repeated_mark",
    "normalized_changed",
    "norm_edit_distance",
    "in_correct_dict",
    "in_error_dict",
    "mfr_exists",
    "mfr_changed",
    "mfr_conf",
    "change_rate",
    "raw_total_log",
    "error_candidate_count",
    "rule_candidate_count",
    "prev_has_thai",
    "next_has_thai",
    "prev_changed_mfr",
    "next_changed_mfr",
    "position_frac",
    "sent_len_log",
]


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def is_thai_char(ch: str) -> bool:
    return "\u0E00" <= ch <= "\u0E7F"


def has_thai(text: str) -> bool:
    return bool(THAI_RE.search(text or ""))


def thai_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if is_thai_char(c)) / max(1, len(text))


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ("A" <= c <= "Z") or ("a" <= c <= "z")) / max(1, len(text))


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / max(1, len(text))


def is_url_like(text: str) -> bool:
    return bool(URL_RE.search(text or ""))


def is_protected_token(text: str) -> bool:
    """Tokens that should be changed only with very strong evidence."""
    if not text:
        return True
    if text.startswith("#") or text.startswith("@"):
        return True
    if is_url_like(text):
        return True
    # Mostly Latin / numeric identifiers are often names, codes, English words.
    if not has_thai(text) and (LATIN_RE.search(text) or DIGIT_RE.search(text)):
        return True
    if DIGIT_RE.search(text) and latin_ratio(text) + digit_ratio(text) > 0.5:
        return True
    return False


def collapse_same_marks(text: str) -> str:
    if not text:
        return text
    out = []
    for ch in text:
        if out and ch == out[-1] and ch in THAI_MARKS:
            continue
        out.append(ch)
    return "".join(out)


def compress_repeated_chars(text: str, max_repeat: int = 2) -> str:
    if not text:
        return text
    out: List[str] = []
    prev = None
    cnt = 0
    for ch in text:
        if ch == prev:
            cnt += 1
        else:
            prev = ch
            cnt = 1
        if cnt <= max_repeat:
            out.append(ch)
    return "".join(out)


def compress_repeated_chars_to_one(text: str) -> str:
    if not text:
        return text
    out: List[str] = []
    prev = None
    for ch in text:
        if ch == prev:
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def _replace_double_leading_e(text: str) -> str:
    # Two consecutive Thai leading vowel เ visually correspond to แ in the XNCC routine.
    return text.replace("เเ", "แ")


def _combine_sara_am(text: str) -> str:
    # NFKC may decompose SARA AM. Put it back for stable token comparison.
    return text.replace("\u0e4d\u0e32", "\u0e33")


def _remove_zero_width(text: str) -> str:
    return text.translate({ord(c): None for c in ZERO_WIDTH})


def reorder_thai_marks_once(text: str) -> str:
    """Apply conservative Thai mark-order fixes.

    The intended pattern is roughly L C H T F. This function only fixes
    obvious local inversions and never inserts/deletes base characters.
    """
    if not text:
        return text
    L = re.escape(LEADING_VOWELS)
    T = re.escape(TONE_MARKS)
    H = re.escape(HANGING_VOWELS)
    F = re.escape(FOLLOWING_VOWELS)
    C = re.escape(BASE_THAI_CHARS)
    text = re.sub(f"([{L}])([{T}])([{C}])", r"\1\3\2", text)  # L T C -> L C T
    text = re.sub(f"([{C}])([{T}])([{H}])", r"\1\3\2", text)  # C T H -> C H T
    text = re.sub(f"([{C}])([{F}])([{T}])", r"\1\3\2", text)  # C F T -> C T F
    return text


def thai_normalize_token(text: str) -> str:
    """Stable, non-destructive Thai Unicode/mark normalization.

    This is designed for candidate generation. In submission, the system still
    uses a detector/reranker before accepting the normalized token.
    """
    if text is None:
        return ""
    x = str(text)
    x = unicodedata.normalize("NFKC", x)
    x = _remove_zero_width(x)
    x = _replace_double_leading_e(x)
    x = _combine_sara_am(x)
    x = collapse_same_marks(x)
    prev = None
    for _ in range(3):
        if x == prev:
            break
        prev = x
        x = reorder_thai_marks_once(x)
        x = collapse_same_marks(x)
        x = _combine_sara_am(x)
    return x


def edit_distance(a: str, b: str, max_distance: Optional[int] = None) -> int:
    if a == b:
        return 0
    if a is None:
        a = ""
    if b is None:
        b = ""
    if len(a) < len(b):
        a, b = b, a
    if max_distance is not None and len(a) - len(b) > max_distance:
        return max_distance + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(val)
            row_min = min(row_min, val)
        prev = cur
        if max_distance is not None and row_min > max_distance:
            return max_distance + 1
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> int:
    return edit_distance(thai_normalize_token(a), thai_normalize_token(b))


def thai_rule_candidates(raw: str, max_candidates: int = 12) -> List[str]:
    """Generate conservative Thai-specific candidates."""
    if raw is None:
        return []
    raw = str(raw)
    cands: List[str] = []

    def add(x: str) -> None:
        if x and x != raw and x not in cands and not SPACE_RE.search(x):
            cands.append(x)

    n = thai_normalize_token(raw)
    add(n)
    add(collapse_same_marks(raw))
    add(_replace_double_leading_e(raw))
    add(_combine_sara_am(raw))
    add(reorder_thai_marks_once(raw))

    if re.search(r"(.)\1{2,}", raw):
        add(compress_repeated_chars(raw, 2))
        add(compress_repeated_chars_to_one(raw))
        add(thai_normalize_token(compress_repeated_chars(raw, 2)))
        add(thai_normalize_token(compress_repeated_chars_to_one(raw)))

    # Only keep candidates that still resemble the original script profile.
    filtered = []
    for cand in cands:
        if has_thai(raw) and not has_thai(cand):
            continue
        if len(cand) > max(1, len(raw) * 2 + 3):
            continue
        filtered.append(cand)
    return filtered[:max_candidates]


def target_or_raw(raw: str, norm: Optional[str]) -> str:
    return raw if norm is None else norm


@dataclass
class ResourceConfig:
    min_count: int = 1
    min_mfr_count: int = 1


def build_resources(rows: Iterable[Dict[str, Any]], lang: str = "th", config: Optional[ResourceConfig] = None) -> Dict[str, Any]:
    config = config or ResourceConfig()
    counts: Dict[str, Counter] = defaultdict(Counter)
    correct_counter: Counter = Counter()
    raw_counter: Counter = Counter()
    error_dict: Dict[str, Counter] = defaultdict(Counter)
    pair_counter: Counter = Counter()

    n_rows = 0
    n_tokens = 0
    n_changed = 0
    for row in rows:
        if row.get("lang") != lang:
            continue
        raw_words = row.get("raw", [])
        norm_words = row.get("norm", raw_words)
        n_rows += 1
        for raw, norm in zip(raw_words, norm_words):
            raw = str(raw)
            gold = target_or_raw(raw, norm)
            counts[raw][gold] += 1
            raw_counter[raw] += 1
            correct_counter[gold] += 1
            pair_counter[(raw, gold)] += 1
            n_tokens += 1
            if raw != gold:
                n_changed += 1
                error_dict[raw][gold] += 1

    mfr: Dict[str, str] = {}
    mfr_stats: Dict[str, Dict[str, float]] = {}
    for raw, ctr in counts.items():
        total = sum(ctr.values())
        best, best_count = max(ctr.items(), key=lambda x: (x[1], x[0] == raw, x[0]))
        changed_count = total - ctr.get(raw, 0)
        mfr[raw] = best
        mfr_stats[raw] = {
            "total": float(total),
            "best_count": float(best_count),
            "conf": float(best_count / max(1, total)),
            "change_rate": float(changed_count / max(1, total)),
            "changed": float(best != raw),
        }

    # Convert Counters to plain dicts for stable pickling / JSON-like inspection.
    resources = {
        "lang": lang,
        "n_rows": n_rows,
        "n_tokens": n_tokens,
        "n_changed": n_changed,
        "counts": {k: dict(v) for k, v in counts.items()},
        "mfr": mfr,
        "mfr_stats": mfr_stats,
        "correct_counter": dict(correct_counter),
        "raw_counter": dict(raw_counter),
        "correct_set": set(correct_counter.keys()),
        "error_dict": {k: dict(v) for k, v in error_dict.items()},
        "pair_counter": dict(pair_counter),
    }
    return resources


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_mfr_info(resources: Dict[str, Any], raw: str) -> Tuple[str, Dict[str, float]]:
    mfr = resources.get("mfr", {})
    stats = resources.get("mfr_stats", {})
    return mfr.get(raw, raw), stats.get(raw, {"total": 0.0, "conf": 0.0, "change_rate": 0.0, "changed": 0.0, "best_count": 0.0})


def extract_features(raw: str, prev_tok: str, next_tok: str, pos: int, sent_len: int, resources: Dict[str, Any]) -> List[float]:
    raw = "" if raw is None else str(raw)
    prev_tok = "" if prev_tok is None else str(prev_tok)
    next_tok = "" if next_tok is None else str(next_tok)
    L = len(raw)
    normed = thai_normalize_token(raw)
    mfr_best, st = get_mfr_info(resources, raw)
    correct_set = resources.get("correct_set", set())
    error_dict = resources.get("error_dict", {})
    rule_cands = thai_rule_candidates(raw)
    prev_mfr, prev_st = get_mfr_info(resources, prev_tok)
    next_mfr, next_st = get_mfr_info(resources, next_tok)

    num_tone = sum(raw.count(ch) for ch in TONE_MARKS)
    num_hang = sum(raw.count(ch) for ch in HANGING_VOWELS)
    num_lead = sum(raw.count(ch) for ch in LEADING_VOWELS)
    num_follow = sum(raw.count(ch) for ch in FOLLOWING_VOWELS)
    has_rep = bool(re.search(r"(.)\1{1,}", raw))
    has_rep_mark = bool(re.search(f"([{re.escape(THAI_MARKS)}])\\1+", raw))
    tr = thai_ratio(raw)
    lr = latin_ratio(raw)
    dr = digit_ratio(raw)

    return [
        float(L),
        math.log1p(float(st.get("total", 0.0))),
        tr,
        lr,
        dr,
        float(tr > 0),
        float(bool(LATIN_RE.search(raw))),
        float(bool(DIGIT_RE.search(raw))),
        float(is_url_like(raw)),
        float(raw.startswith("#")),
        float(raw.startswith("@")),
        float(tr > 0 and tr >= 0.95),
        float((tr > 0 and (lr > 0 or dr > 0))),
        float(num_tone),
        float(num_hang),
        float(num_lead),
        float(num_follow),
        float(has_rep),
        float(has_rep_mark),
        float(normed != raw),
        float(edit_distance(raw, normed, max_distance=6)),
        float(raw in correct_set),
        float(raw in error_dict),
        float(raw in resources.get("mfr", {})),
        float(mfr_best != raw),
        float(st.get("conf", 0.0)),
        float(st.get("change_rate", 0.0)),
        math.log1p(float(st.get("total", 0.0))),
        float(len(error_dict.get(raw, {}))),
        float(len(rule_cands)),
        float(has_thai(prev_tok)),
        float(has_thai(next_tok)),
        float(prev_mfr != prev_tok and prev_st.get("conf", 0.0) >= 0.8),
        float(next_mfr != next_tok and next_st.get("conf", 0.0) >= 0.8),
        float(pos / max(1, sent_len - 1)) if sent_len > 1 else 0.0,
        math.log1p(float(sent_len)),
    ]


def rows_to_xy(rows: Sequence[Dict[str, Any]], resources: Dict[str, Any], lang: str = "th") -> Tuple[List[List[float]], List[int], List[Tuple[str, str]]]:
    X: List[List[float]] = []
    y: List[int] = []
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        if row.get("lang") != lang:
            continue
        raw_words = row.get("raw", [])
        norm_words = row.get("norm", raw_words)
        sent_len = len(raw_words)
        for i, raw in enumerate(raw_words):
            gold = target_or_raw(raw, norm_words[i] if i < len(norm_words) else raw)
            prev_tok = raw_words[i - 1] if i > 0 else ""
            next_tok = raw_words[i + 1] if i + 1 < sent_len else ""
            X.append(extract_features(raw, prev_tok, next_tok, i, sent_len, resources))
            y.append(int(str(raw) != str(gold)))
            pairs.append((str(raw), str(gold)))
    return X, y, pairs


def simple_non_byt5_candidate(
    raw: str,
    resources: Dict[str, Any],
    min_mfr_conf: float = 0.80,
    min_mfr_count: int = 2,
    max_rule_candidates: int = 8,
) -> Tuple[str, str, float]:
    """Best non-neural candidate for threshold tuning / fallback.

    Returns: (candidate, source, score)
    """
    raw = str(raw)
    best = raw
    best_source = "keep"
    best_score = 0.0
    mfr_best, st = get_mfr_info(resources, raw)
    if mfr_best != raw and st.get("conf", 0.0) >= min_mfr_conf and st.get("total", 0.0) >= min_mfr_count:
        best = mfr_best
        best_source = "mfr"
        best_score = 3.0 + 3.0 * st.get("conf", 0.0) + math.log1p(st.get("total", 0.0)) / 4.0

    err_map = resources.get("error_dict", {}).get(raw, {})
    if err_map:
        total = sum(err_map.values())
        cand, cnt = max(err_map.items(), key=lambda x: (x[1], x[0] == mfr_best, x[0]))
        score = 2.8 + 2.0 * (cnt / max(1, total)) + math.log1p(cnt) / 4.0
        if score > best_score:
            best, best_source, best_score = cand, "errdict", score

    correct_set = resources.get("correct_set", set())
    for cand in thai_rule_candidates(raw, max_candidates=max_rule_candidates):
        score = 1.3 + float(cand in correct_set) + float(cand == mfr_best)
        score -= 0.15 * edit_distance(raw, cand, max_distance=8)
        if score > best_score:
            best, best_source, best_score = cand, "rule", score

    return best, best_source, best_score


def evaluate_token_level(raw_list: List[List[str]], gold_list: List[List[str]], pred_list: List[List[str]], info: bool = True, prefix: str = "") -> Tuple[float, float, float]:
    total = 0
    changed = 0
    cor = 0
    false_positive = 0
    true_change_pred = 0
    missed_change = 0
    for raw_sent, gold_sent, pred_sent in zip(raw_list, gold_list, pred_list):
        for r, g, p in zip(raw_sent, gold_sent, pred_sent):
            total += 1
            if r != g:
                changed += 1
            if p == g:
                cor += 1
            if r == g and p != r:
                false_positive += 1
            if r != g and p != r:
                true_change_pred += 1
            if r != g and p == r:
                missed_change += 1
    acc = cor / max(1, total)
    lai = (total - changed) / max(1, total)
    err = (acc - lai) / max(1e-12, (1.0 - lai)) if changed > 0 else 0.0
    if info:
        print(f"{prefix}Baseline acc.(LAI): {lai * 100:.2f}")
        print(f"{prefix}Accuracy:           {acc * 100:.2f}")
        print(f"{prefix}ERR:                {err * 100:.2f}")
        print(f"{prefix}Changed gold:       {changed}/{total} ({changed / max(1,total) * 100:.2f}%)")
        print(f"{prefix}False positives:    {false_positive}")
        print(f"{prefix}Changed predicted: {sum(1 for rs, ps in zip(raw_list, pred_list) for r,p in zip(rs,ps) if r != p)}")
        print(f"{prefix}Missed changes:     {missed_change}")
    return lai, acc, err


def collect_raw_gold(rows: Sequence[Dict[str, Any]], lang: str = "th") -> Tuple[List[List[str]], List[List[str]]]:
    raw_list: List[List[str]] = []
    gold_list: List[List[str]] = []
    for row in rows:
        if row.get("lang") != lang:
            continue
        raw_words = [str(x) for x in row.get("raw", [])]
        norm_words = row.get("norm")
        if norm_words is None:
            continue
        gold_words = [target_or_raw(r, n) for r, n in zip(raw_words, norm_words)]
        raw_list.append(raw_words)
        gold_list.append(gold_words)
    return raw_list, gold_list


def build_mfr_for_any_lang(rows: Iterable[Dict[str, Any]], lang: str) -> Dict[str, str]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.get("lang") != lang:
            continue
        for r, n in zip(row.get("raw", []), row.get("norm", row.get("raw", []))):
            target = target_or_raw(str(r), n)
            counts[str(r)][target] += 1
    return {r: max(c.items(), key=lambda x: (x[1], x[0] == r, x[0]))[0] for r, c in counts.items()}
