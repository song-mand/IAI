import argparse
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
IT_LANG = "it"

TARGET_LANGS = sorted([
    "en", "da", "de", "es", "hr", "it", "nl", "sl", "sr", "tr",
    "iden", "trde", "id", "ja", "ko", "th", "vi"
])


URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
MENTION_RE = re.compile(r"^@\w+")
HASHTAG_RE = re.compile(r"^#\S+")
PUNCT_ONLY_RE = re.compile(r"^[^\wÀ-ÖØ-öø-ÿ]+$")
REPEAT_RE = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ])\1{2,}")


SMS_ABBREV_SET = {
    "nn", "nnt", "cmq", "cqm", "ke", "ki", "x", "xke", "xké", "xkè",
    "xche", "xchè", "xché", "perke", "perké", "perkè", "grz", "qnd",
    "cn", "dv", "qst", "qsto", "qsta", "qsti", "qste", "tt", "sn",
    "sx", "dx", "info", "nov",
}


def safe_norm(raw: str, norm: Any) -> str:
    return raw if norm is None else str(norm)


def get_lang_rows(split, lang: str) -> List[Dict[str, Any]]:
    return [row for row in split if row["lang"] == lang]


def strip_accents_and_marks(s: str) -> str:
    s = str(s)
    s = s.replace("’", "'").replace("`", "'").replace("´", "'")
    s = s.replace("'", "")

    decomposed = unicodedata.normalize("NFD", s)
    without_accents = "".join(
        ch for ch in decomposed
        if unicodedata.category(ch) != "Mn"
    )

    return without_accents.lower()


def is_protected_token(tok: str) -> bool:
    return (
        bool(URL_RE.match(tok))
        or bool(MENTION_RE.match(tok))
        or bool(HASHTAG_RE.match(tok))
        or bool(PUNCT_ONLY_RE.match(tok))
    )


def build_mfr_counts(rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    counts = defaultdict(Counter)

    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            raw = str(raw)
            norm = safe_norm(raw, norm)
            counts[raw][norm] += 1

    return counts


def choose_mfr(raw: str, counter: Counter) -> str:
    if not counter:
        return raw

    max_count = max(counter.values())
    bests = [k for k, v in counter.items() if v == max_count]

    if raw in bests:
        return raw

    return sorted(bests)[0]


def build_mfr_dict(mfr_counts: Dict[str, Counter]) -> Dict[str, str]:
    return {
        raw: choose_mfr(raw, counter)
        for raw, counter in mfr_counts.items()
    }


def get_mapping_stats(raw: str, cand: str, mfr_counts: Dict[str, Counter]) -> Tuple[int, int, float]:
    counter = mfr_counts.get(raw, Counter())
    total = sum(counter.values())
    count = counter.get(cand, 0)
    conf = count / total if total else 0.0
    return count, total, conf


def is_case_only(raw: str, cand: str) -> bool:
    return raw.lower() == cand.lower() and raw != cand


def is_diacritic_mapping(raw: str, cand: str) -> bool:
    if raw == cand:
        return False

    # 대소문자만 다른 경우는 diacritic이 아니라 case category로 보내야 함
    if raw.lower() == cand.lower():
        return False

    raw_can = strip_accents_and_marks(raw)
    cand_can = strip_accents_and_marks(cand)

    return raw_can == cand_can

def is_repeat_mapping(raw: str, cand: str) -> bool:
    if REPEAT_RE.search(raw) is None:
        return False

    reduced_one = REPEAT_RE.sub(r"\1", raw)
    reduced_two = REPEAT_RE.sub(r"\1\1", raw)

    return cand in {reduced_one, reduced_two}


def classify_mapping(raw: str, cand: str) -> str:
    if raw == cand:
        return "copy"

    if is_protected_token(raw):
        return "protected"

    if " " in cand:
        return "multiword"

    # 중요: case-only를 diacritic보다 먼저 판정해야 함
    if is_case_only(raw, cand):
        if raw.isupper() and cand.islower():
            return "allcaps_lower"
        if raw.isupper() and cand[:1].isupper() and cand[1:].islower():
            return "allcaps_title"
        if raw[:1].isupper() and cand.islower():
            return "title_lower"
        if raw.islower() and cand[:1].isupper():
            return "lower_title"
        return "case_only"

    if is_diacritic_mapping(raw, cand):
        return "diacritic"

    if is_repeat_mapping(raw, cand):
        return "repeat"

    raw_low = raw.lower()

    if raw_low in SMS_ABBREV_SET:
        return "sms_abbrev"

    if len(cand) >= len(raw) + 4:
        return "expansion"

    if len(raw) <= 3 and len(cand) > len(raw):
        return "short_expansion"

    return "spelling_or_other"

def compute_loo_reliability(rows: List[Dict[str, Any]], mfr_counts: Dict[str, Counter]) -> Dict[str, Any]:
    """
    Leave-One-Out Reliability 계산.

    각 train token occurrence를 하나씩 test처럼 간주하고,
    그 occurrence를 제외한 나머지 train으로 MFR을 만들었을 때
    gold를 맞히는지 측정한다.
    """
    raw_total = Counter()
    raw_correct = Counter()

    pair_total = Counter()
    pair_correct = Counter()

    category_total = Counter()
    category_correct = Counter()

    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            raw = str(raw)
            gold = safe_norm(raw, norm)

            loo_counter = Counter(mfr_counts.get(raw, Counter()))
            loo_counter[gold] -= 1

            if loo_counter[gold] <= 0:
                del loo_counter[gold]

            loo_pred = choose_mfr(raw, loo_counter)

            raw_total[raw] += 1
            if loo_pred == gold:
                raw_correct[raw] += 1

            if loo_pred != raw:
                category = classify_mapping(raw, loo_pred)

                category_total[category] += 1
                if loo_pred == gold:
                    category_correct[category] += 1

                pair_key = (raw, loo_pred)
                pair_total[pair_key] += 1
                if loo_pred == gold:
                    pair_correct[pair_key] += 1

    raw_stats = {}
    for raw, total in raw_total.items():
        correct = raw_correct[raw]
        raw_stats[raw] = {
            "total": total,
            "correct": correct,
            "precision": correct / total if total else 0.0,
        }

    pair_stats = {}
    for pair_key, total in pair_total.items():
        correct = pair_correct[pair_key]
        raw, cand = pair_key
        pair_stats[f"{raw}\t{cand}"] = {
            "raw": raw,
            "candidate": cand,
            "total": total,
            "correct": correct,
            "precision": correct / total if total else 0.0,
            "category": classify_mapping(raw, cand),
        }

    category_stats = {}
    for category, total in category_total.items():
        correct = category_correct[category]
        category_stats[category] = {
            "total": total,
            "correct": correct,
            "precision": correct / total if total else 0.0,
        }

    return {
        "raw_stats": raw_stats,
        "pair_stats": pair_stats,
        "category_stats": category_stats,
    }


def get_raw_loo(raw: str, loo: Dict[str, Any]) -> Tuple[int, float]:
    stat = loo["raw_stats"].get(raw)
    if not stat:
        return 0, 0.0
    return stat["total"], stat["precision"]


def get_pair_loo(raw: str, cand: str, loo: Dict[str, Any]) -> Tuple[int, float]:
    stat = loo["pair_stats"].get(f"{raw}\t{cand}")
    if not stat:
        return 0, 0.0
    return stat["total"], stat["precision"]


def get_category_loo(category: str, loo: Dict[str, Any]) -> Tuple[int, float]:
    stat = loo["category_stats"].get(category)
    if not stat:
        return 0, 0.0
    return stat["total"], stat["precision"]


POLICIES = {
    "l0_mfr": {
        "description": "plain MFR baseline",
        "thresholds": None,
    },

    "l1_loo_raw": {
        "description": "MFR with raw-level LOO reliability gate",
        "default": {
            "min_count": 2,
            "min_conf": 0.60,
            "min_raw_loo_total": 3,
            "min_raw_loo": 0.60,
            "min_pair_loo_total": 2,
            "min_pair_loo": 0.50,
            "min_cat_loo_total": 5,
            "min_cat_loo": 0.50,
        },
        "thresholds": {},
    },

    "l2_loo_category": {
        "description": "MFR with category-specific LOO reliability gate",
        "default": {
            "min_count": 3,
            "min_conf": 0.80,
            "min_raw_loo_total": 3,
            "min_raw_loo": 0.60,
            "min_pair_loo_total": 2,
            "min_pair_loo": 0.60,
            "min_cat_loo_total": 5,
            "min_cat_loo": 0.60,
        },
        "thresholds": {
            "diacritic": {
                "min_count": 1,
                "min_conf": 0.50,
                "min_raw_loo_total": 999999,
                "min_raw_loo": 0.00,
                "min_pair_loo_total": 999999,
                "min_pair_loo": 0.00,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.50,
            },
            "sms_abbrev": {
                "min_count": 2,
                "min_conf": 0.70,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.55,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.55,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.55,
            },
            "repeat": {
                "min_count": 2,
                "min_conf": 0.75,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.55,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.55,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.55,
            },
            "allcaps_lower": {
                "min_count": 3,
                "min_conf": 0.90,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.70,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.70,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.70,
            },
            "allcaps_title": {
                "min_count": 3,
                "min_conf": 0.85,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.70,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.70,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.70,
            },
            "title_lower": {
                "min_count": 4,
                "min_conf": 0.85,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.70,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.70,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.70,
            },
            "lower_title": {
                "min_count": 4,
                "min_conf": 0.85,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.70,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.70,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.70,
            },
            "expansion": {
                "min_count": 4,
                "min_conf": 0.90,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.70,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.70,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.70,
            },
            "short_expansion": {
                "min_count": 3,
                "min_conf": 0.80,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.60,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.60,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.60,
            },
            "spelling_or_other": {
                "min_count": 3,
                "min_conf": 0.90,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.70,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.70,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.70,
            },
        },
    },

    "l3_cap_strict": {
        "description": "LOO reliability, strict on capitalization; allows diacritic/abbrev/repeat",
        "default": {
            "min_count": 3,
            "min_conf": 0.90,
            "min_raw_loo_total": 3,
            "min_raw_loo": 0.70,
            "min_pair_loo_total": 2,
            "min_pair_loo": 0.70,
            "min_cat_loo_total": 5,
            "min_cat_loo": 0.70,
        },
        "thresholds": {
            "diacritic": {
                "min_count": 1,
                "min_conf": 0.50,
                "min_raw_loo_total": 999999,
                "min_raw_loo": 0.00,
                "min_pair_loo_total": 999999,
                "min_pair_loo": 0.00,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.50,
            },
            "sms_abbrev": {
                "min_count": 2,
                "min_conf": 0.75,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.60,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.60,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.60,
            },
            "repeat": {
                "min_count": 2,
                "min_conf": 0.80,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.60,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.60,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.60,
            },
            # capitalization 계열은 매우 엄격하게 일부만 허용.
            "allcaps_title": {
                "min_count": 5,
                "min_conf": 0.90,
                "min_raw_loo_total": 3,
                "min_raw_loo": 0.80,
                "min_pair_loo_total": 2,
                "min_pair_loo": 0.80,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.80,
            },
        },
    },

    "l4_accent_only": {
        "description": "only diacritic mappings with LOO category support",
        "default": {},
        "thresholds": {
            "diacritic": {
                "min_count": 1,
                "min_conf": 0.50,
                "min_raw_loo_total": 999999,
                "min_raw_loo": 0.00,
                "min_pair_loo_total": 999999,
                "min_pair_loo": 0.00,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.50,
            },
        },
    },

    "l5_high_precision": {
        "description": "very strict LOO reliability gate",
        "default": {
            "min_count": 3,
            "min_conf": 0.90,
            "min_raw_loo_total": 2,
            "min_raw_loo": 0.80,
            "min_pair_loo_total": 2,
            "min_pair_loo": 0.80,
            "min_cat_loo_total": 5,
            "min_cat_loo": 0.75,
        },
        "thresholds": {
            "diacritic": {
                "min_count": 1,
                "min_conf": 0.60,
                "min_raw_loo_total": 999999,
                "min_raw_loo": 0.00,
                "min_pair_loo_total": 999999,
                "min_pair_loo": 0.00,
                "min_cat_loo_total": 5,
                "min_cat_loo": 0.60,
            },
        },
    },
}


def get_thresholds(mode: str, category: str) -> Dict[str, Any]:
    policy = POLICIES[mode]

    base = dict(policy.get("default", {}))
    override = policy.get("thresholds", {}).get(category, None)

    if override is None:
        return base

    base.update(override)
    return base


def check_threshold(value_total: int, value_precision: float, min_total: int, min_precision: float) -> bool:
    """
    total이 충분히 클 때만 precision threshold를 적용한다.
    total이 min_total보다 작으면 reliability 정보가 부족하므로 통과시킨다.
    count/conf/category threshold가 별도로 통제한다.
    """
    if value_total < min_total:
        return True

    return value_precision >= min_precision


def allow_mapping(
    raw: str,
    cand: str,
    mfr_counts: Dict[str, Counter],
    loo: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    if raw == cand:
        return {
            "allowed": True,
            "category": "copy",
            "reason": "copy",
            "count": 0,
            "total": 0,
            "confidence": 1.0,
            "raw_loo_total": 0,
            "raw_loo": 1.0,
            "pair_loo_total": 0,
            "pair_loo": 1.0,
            "category_loo_total": 0,
            "category_loo": 1.0,
        }

    category = classify_mapping(raw, cand)
    count, total, conf = get_mapping_stats(raw, cand, mfr_counts)

    raw_loo_total, raw_loo = get_raw_loo(raw, loo)
    pair_loo_total, pair_loo = get_pair_loo(raw, cand, loo)
    cat_loo_total, cat_loo = get_category_loo(category, loo)

    result = {
        "allowed": False,
        "category": category,
        "reason": "",
        "count": count,
        "total": total,
        "confidence": conf,
        "raw_loo_total": raw_loo_total,
        "raw_loo": raw_loo,
        "pair_loo_total": pair_loo_total,
        "pair_loo": pair_loo,
        "category_loo_total": cat_loo_total,
        "category_loo": cat_loo,
    }

    if mode == "l0_mfr":
        result["allowed"] = True
        result["reason"] = "plain_mfr"
        return result

    if category == "protected":
        result["reason"] = "protected"
        return result

    thresholds = get_thresholds(mode, category)

    if not thresholds:
        result["reason"] = f"category_not_allowed:{category}"
        return result

    if count < thresholds.get("min_count", 1):
        result["reason"] = "count_below_threshold"
        return result

    if conf < thresholds.get("min_conf", 0.0):
        result["reason"] = "confidence_below_threshold"
        return result

    if not check_threshold(
        raw_loo_total,
        raw_loo,
        thresholds.get("min_raw_loo_total", 999999),
        thresholds.get("min_raw_loo", 0.0),
    ):
        result["reason"] = "raw_loo_below_threshold"
        return result

    if not check_threshold(
        pair_loo_total,
        pair_loo,
        thresholds.get("min_pair_loo_total", 999999),
        thresholds.get("min_pair_loo", 0.0),
    ):
        result["reason"] = "pair_loo_below_threshold"
        return result

    if not check_threshold(
        cat_loo_total,
        cat_loo,
        thresholds.get("min_cat_loo_total", 999999),
        thresholds.get("min_cat_loo", 0.0),
    ):
        result["reason"] = "category_loo_below_threshold"
        return result

    result["allowed"] = True
    result["reason"] = "passed"
    return result


def predict_it_sentence(
    raw_words: List[str],
    mfr_dict: Dict[str, str],
    mfr_counts: Dict[str, Counter],
    loo: Dict[str, Any],
    mode: str,
    debug_collector: List[Dict[str, Any]] = None,
) -> List[str]:
    pred = []

    for i, raw in enumerate(raw_words):
        raw = str(raw)
        cand = mfr_dict.get(raw, raw)

        decision = allow_mapping(
            raw=raw,
            cand=cand,
            mfr_counts=mfr_counts,
            loo=loo,
            mode=mode,
        )

        final = cand if decision["allowed"] else raw
        pred.append(final)

        if debug_collector is not None and cand != raw:
            debug_collector.append({
                "raw": raw,
                "mfr": cand,
                "final": final,
                "allowed": decision["allowed"],
                "category": decision["category"],
                "reason": decision["reason"],
                "count": decision["count"],
                "total": decision["total"],
                "confidence": decision["confidence"],
                "raw_loo_total": decision["raw_loo_total"],
                "raw_loo": decision["raw_loo"],
                "pair_loo_total": decision["pair_loo_total"],
                "pair_loo": decision["pair_loo"],
                "category_loo_total": decision["category_loo_total"],
                "category_loo": decision["category_loo"],
                "prev": raw_words[i - 1] if i > 0 else "<BOS>",
                "next": raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>",
                "sentence": raw_words,
            })

    return pred


def predict_mfr_sentence(raw_words: List[str], mfr_dict: Dict[str, str]) -> List[str]:
    return [mfr_dict.get(str(w), str(w)) for w in raw_words]


def analyze_predictions(
    lang: str,
    rows: List[Dict[str, Any]],
    pred_rows: List[List[str]],
    debug_items: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total = 0
    changed = 0

    change_counter = Counter()
    allowed_category_counter = Counter()
    blocked_category_counter = Counter()
    blocked_reason_counter = Counter()
    samples = []

    for row, pred in zip(rows, pred_rows):
        raw_words = [str(x) for x in row["raw"]]
        changes = []

        for raw, p in zip(raw_words, pred):
            total += 1

            if raw != p:
                changed += 1
                change_counter[(raw, p)] += 1
                changes.append((raw, p))

        if changes and len(samples) < 30:
            samples.append({
                "raw": raw_words,
                "pred": pred,
                "changes": changes,
            })

    if debug_items:
        for item in debug_items:
            if item["allowed"]:
                allowed_category_counter[item["category"]] += 1
            else:
                blocked_category_counter[item["category"]] += 1
                blocked_reason_counter[item["reason"]] += 1

    return {
        "lang": lang,
        "total_tokens": total,
        "changed_tokens": changed,
        "changed_rate": changed / total if total else 0.0,
        "top_changes": [
            {"raw": r, "pred": p, "count": c}
            for (r, p), c in change_counter.most_common(100)
        ],
        "allowed_category_counts": dict(allowed_category_counter),
        "blocked_category_counts": dict(blocked_category_counter),
        "blocked_reason_counts": dict(blocked_reason_counter),
        "samples": samples,
    }


def summarize_loo(loo: Dict[str, Any]) -> Dict[str, Any]:
    raw_items = []
    for raw, stat in loo["raw_stats"].items():
        raw_items.append({
            "raw": raw,
            "total": stat["total"],
            "correct": stat["correct"],
            "precision": stat["precision"],
        })

    raw_items.sort(key=lambda x: (x["precision"], -x["total"], x["raw"]))

    pair_items = list(loo["pair_stats"].values())
    pair_items.sort(key=lambda x: (x["precision"], -x["total"], x["raw"]))

    category_items = []
    for category, stat in loo["category_stats"].items():
        category_items.append({
            "category": category,
            "total": stat["total"],
            "correct": stat["correct"],
            "precision": stat["precision"],
        })

    category_items.sort(key=lambda x: x["category"])

    return {
        "category_reliability": category_items,
        "low_raw_reliability": raw_items[:100],
        "low_pair_reliability": pair_items[:100],
    }


def print_analysis(analysis: Dict[str, Any], detail: bool = False):
    print(
        f"[{analysis['lang'].upper():5s}] "
        f"tokens={analysis['total_tokens']:6d} "
        f"changed={analysis['changed_tokens']:5d} "
        f"rate={analysis['changed_rate'] * 100:6.2f}%"
    )

    if not detail:
        return

    print("\n[IT] Top changes")
    for item in analysis["top_changes"][:80]:
        print(f"  {item['raw']} -> {item['pred']} | count={item['count']}")

    print("\n[IT] Allowed category counts")
    print(json.dumps(analysis["allowed_category_counts"], ensure_ascii=False, indent=2))

    print("\n[IT] Blocked category counts")
    print(json.dumps(analysis["blocked_category_counts"], ensure_ascii=False, indent=2))

    print("\n[IT] Blocked reason counts")
    print(json.dumps(analysis["blocked_reason_counts"], ensure_ascii=False, indent=2))

    print("\n[IT] Changed samples")
    for i, sample in enumerate(analysis["samples"][:15], start=1):
        print(f"\n  sample {i}")
        print("  raw : " + " ".join(map(str, sample["raw"])))
        print("  pred: " + " ".join(map(str, sample["pred"])))
        print(f"  changes: {sample['changes']}")


def print_loo_summary(loo_summary: Dict[str, Any]):
    print("\n[IT] LOO category reliability")
    for item in loo_summary["category_reliability"]:
        print(
            f"  {item['category']:18s} "
            f"precision={item['precision']:.3f} "
            f"correct={item['correct']}/{item['total']}"
        )


def print_it_debug(debug_items: List[Dict[str, Any]], max_items: int = 80):
    print("\n[IT] Allowed MFR mappings")
    allowed = [x for x in debug_items if x["allowed"]]
    blocked = [x for x in debug_items if not x["allowed"]]

    for item in allowed[:max_items]:
        print(
            f"  ALLOW {item['raw']} -> {item['final']} "
            f"| cat={item['category']} "
            f"| count={item['count']}/{item['total']} "
            f"| conf={item['confidence']:.3f} "
            f"| raw_loo={item['raw_loo']:.3f}({item['raw_loo_total']}) "
            f"| pair_loo={item['pair_loo']:.3f}({item['pair_loo_total']}) "
            f"| cat_loo={item['category_loo']:.3f}({item['category_loo_total']})"
        )

    print("\n[IT] Blocked MFR mappings")
    for item in blocked[:max_items]:
        print(
            f"  BLOCK {item['raw']} -> {item['mfr']} "
            f"| reason={item['reason']} "
            f"| cat={item['category']} "
            f"| count={item['count']}/{item['total']} "
            f"| conf={item['confidence']:.3f} "
            f"| raw_loo={item['raw_loo']:.3f}({item['raw_loo_total']}) "
            f"| pair_loo={item['pair_loo']:.3f}({item['pair_loo_total']}) "
            f"| cat_loo={item['category_loo']:.3f}({item['category_loo_total']})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--mode", default="l2_loo_category", choices=sorted(POLICIES.keys()))
    parser.add_argument("--out-dir", default="scripts/it/loo_outputs")
    parser.add_argument("--zip-name", default=None)
    parser.add_argument("--debug-it", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    os.makedirs(args.out_dir, exist_ok=True)

    if args.zip_name is None:
        args.zip_name = f"submission_{mode}.zip"

    print("=" * 60)
    print("1. Load dataset")
    print("=" * 60)

    ds = load_dataset(args.dataset)
    train_split = ds["train"]
    test_split = ds["test"]

    print(f"mode: {mode}")
    print(f"description: {POLICIES[mode]['description']}")

    print("=" * 60)
    print("2. Build MFR dictionaries")
    print("=" * 60)

    mfr_counts_by_lang = {}
    mfr_dict_by_lang = {}

    for lang in TARGET_LANGS:
        lang_train = get_lang_rows(train_split, lang)
        mfr_counts = build_mfr_counts(lang_train)
        mfr_dict = build_mfr_dict(mfr_counts)

        mfr_counts_by_lang[lang] = mfr_counts
        mfr_dict_by_lang[lang] = mfr_dict

        print(f"[{lang.upper():5s}] MFR entries: {len(mfr_dict)}")

    print("=" * 60)
    print("3. Compute IT LOO reliability")
    print("=" * 60)

    it_train_rows = get_lang_rows(train_split, IT_LANG)
    it_loo = compute_loo_reliability(
        rows=it_train_rows,
        mfr_counts=mfr_counts_by_lang[IT_LANG],
    )
    it_loo_summary = summarize_loo(it_loo)
    print_loo_summary(it_loo_summary)

    print("=" * 60)
    print("4. Predict")
    print("=" * 60)

    predictions = []
    diagnostics = {
        "method": "IT=LOO-reliability-MFR, others=MFR",
        "mode": mode,
        "policy": POLICIES[mode],
        "it_loo_summary": it_loo_summary,
        "languages": {},
    }

    test_langs = sorted(set(row["lang"] for row in test_split))

    for lang in test_langs:
        rows = get_lang_rows(test_split, lang)
        pred_rows = []
        it_debug_items = []

        print("\n" + "-" * 60)

        if lang == IT_LANG:
            print(f"[{lang.upper()}] Predict with LOO-reliability MFR: {mode}")
        else:
            print(f"[{lang.upper()}] Predict with plain MFR")

        print("-" * 60)

        for row in tqdm(rows, desc=f"[{lang.upper()}] predict"):
            raw_words = [str(x) for x in row["raw"]]

            if lang == IT_LANG:
                pred = predict_it_sentence(
                    raw_words=raw_words,
                    mfr_dict=mfr_dict_by_lang[lang],
                    mfr_counts=mfr_counts_by_lang[lang],
                    loo=it_loo,
                    mode=mode,
                    debug_collector=it_debug_items,
                )
            else:
                pred = predict_mfr_sentence(
                    raw_words=raw_words,
                    mfr_dict=mfr_dict_by_lang.get(lang, {}),
                )

            if len(pred) != len(raw_words):
                print(f"[WARN] Length mismatch in {lang}. Fallback to raw copy.")
                pred = raw_words.copy()

            pred_rows.append(pred)

            predictions.append({
                "raw": raw_words,
                "pred": pred,
                "lang": lang,
            })

        analysis = analyze_predictions(
            lang=lang,
            rows=rows,
            pred_rows=pred_rows,
            debug_items=it_debug_items if lang == IT_LANG else None,
        )

        diagnostics["languages"][lang] = analysis

        print_analysis(analysis, detail=(lang == IT_LANG))

        if lang == IT_LANG and args.debug_it:
            diagnostics["it_loo_decision_debug"] = it_debug_items
            print_it_debug(it_debug_items, max_items=100)

    print("=" * 60)
    print("5. Save outputs")
    print("=" * 60)

    pred_json = os.path.join(args.out_dir, f"predictions_{mode}.json")
    diag_json = os.path.join(args.out_dir, f"diagnostics_{mode}.json")
    zip_path = os.path.join(args.out_dir, args.zip_name)

    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_json, arcname="predictions.json")

    print(f"predictions: {pred_json}")
    print(f"diagnostics: {diag_json}")
    print(f"zip:         {zip_path}")

    print("=" * 60)
    print("6. Zip check")
    print("=" * 60)

    with zipfile.ZipFile(zip_path, "r") as zipf:
        for info in zipf.infolist():
            print(f"{info.filename} | {info.file_size} bytes")

    print("Done.")


if __name__ == "__main__":
    main()