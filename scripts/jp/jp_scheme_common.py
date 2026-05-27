#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japanese lexical normalization utilities for MultiLexNorm-style data.

Scheme:
  COPY first -> high-confidence MFR -> context-sensitive ByT5 -> safety filter.

This file intentionally avoids global kana/kanji/NFKC normalization because the
provided Japanese data did not show that as the main phenomenon.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DELETE_TOKEN = "<DELETE>"

# Tokens that were highly context-dependent in the provided Japanese data.
CONTEXT_SENSITIVE_TOKENS = {
    "ん", "って", "て", "で", "、", "。", "…", "...", "・・", "・・・",
    "です", "ます", "た", "だ", "ね", "よ", "か", "な", "わ",
    "〜", "~", "ー", "！", "!", "？", "?", "）", ")", "｣", "」",
}

# Deletion was observed, but should be restricted to noisy/decoration tokens.
DELETE_CANDIDATES = {
    "〜", "~", "！", "!", "ー", "っ", "ッ", "・", "。", "、", "…", "...",
    "w", "ww", "www", "笑", "え", "わ", "し",
}

# Tokens where a sentence-final punctuation decision is often relevant.
PUNCT_END_CANDIDATES = {
    "です", "ます", "た", "だ", "ね", "よ", "か", "な", "わ", "）", ")", "｣", "」", "…", "。", "、"
}

# Do not use a raw-token MFR shortcut for these. Let ByT5 see the context.
MFR_EXCLUDE_TOKENS = CONTEXT_SENSITIVE_TOKENS | {
    "てる",  # can be safe often, but ByT5 learns it well and context is cheap.
    "でる",
}

URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
NUMERIC_RE = re.compile(r"^[+-]?[0-9０-９]+([.,．][0-9０-９]+)?(%|％)?$")
DATE_LIKE_RE = re.compile(r"^[0-9０-９]{1,4}[/-][0-9０-９]{1,2}([/-][0-9０-９]{1,2})?$")
ASCII_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+.-]*$")

HIRAGANA_RE = re.compile(r"[\u3040-\u309F]")
KATAKANA_RE = re.compile(r"[\u30A0-\u30FF]")
KANJI_RE = re.compile(r"[\u4E00-\u9FFF]")
JP_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")


def as_list(x: Any) -> List[str]:
    """Convert parquet/list/np.ndarray cell to a plain list of strings."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    return ["" if v is None else str(v) for v in list(x)]


def normalize_empty_norm(raw_words: Sequence[str], norm_words: Optional[Sequence[str]]) -> List[str]:
    """If norm is missing/empty in test data, return a length-matched empty list."""
    if norm_words is None:
        return ["" for _ in raw_words]
    norm = as_list(norm_words)
    if len(norm) != len(raw_words):
        return ["" for _ in raw_words]
    return norm


def has_japanese(s: str) -> bool:
    return bool(JP_RE.search(s or ""))


def script_signature(s: str) -> str:
    """Coarse script signature for diagnostics/safety."""
    flags = []
    if HIRAGANA_RE.search(s or ""):
        flags.append("hira")
    if KATAKANA_RE.search(s or ""):
        flags.append("kata")
    if KANJI_RE.search(s or ""):
        flags.append("kanji")
    if re.search(r"[A-Za-zＡ-Ｚａ-ｚ]", s or ""):
        flags.append("latin")
    if re.search(r"[0-9０-９]", s or ""):
        flags.append("num")
    return "+".join(flags) if flags else "other"


def is_url_like(token: str) -> bool:
    return bool(URL_RE.search(token or "") or EMAIL_RE.search(token or ""))


def is_numeric_like(token: str) -> bool:
    token = token or ""
    return bool(NUMERIC_RE.match(token) or DATE_LIKE_RE.match(token))


def is_social_protected(token: str) -> bool:
    token = token or ""
    return token.startswith("@") or token.startswith("#") or token.startswith("＃")


def is_hard_keep(token: str) -> bool:
    """
    Tokens that should normally be copied unless they are explicitly captured by
    high-confidence training evidence.
    """
    if token == "":
        return True
    if is_url_like(token) or is_numeric_like(token) or is_social_protected(token):
        return True
    # Pure ASCII words are usually names/platform tokens. Some are real abbreviations
    # such as RT/TL/DM/GW; those can still be handled by high-confidence MFR before
    # this hard keep is applied by the caller.
    if ASCII_ID_RE.match(token) and token.upper() not in {"RT", "TL", "DM", "GW", "OK", "SNS"}:
        return True
    return False


def is_near_sentence_end(words: Sequence[str], i: int) -> bool:
    if i >= len(words) - 1:
        return True
    nxt = words[i + 1]
    if nxt in {"。", "！", "!", "？", "?", "…", "..."}:
        return True
    if nxt in {"」", "｣", "）", ")"} and i + 1 >= len(words) - 1:
        return True
    return False


def make_byt5_input(lang: str, words: Sequence[str], i: int) -> str:
    """Input format used both for training and inference."""
    context = " ".join(words)
    target = words[i]
    return f"normalize {lang}: word: {target} index: {i} context: {context}"


def encode_target(norm: str) -> str:
    return DELETE_TOKEN if norm == "" else norm


def decode_target(pred: str) -> str:
    pred = (pred or "").strip()
    if pred == DELETE_TOKEN:
        return ""
    return pred


def token_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


@dataclass
class ArtifactConfig:
    min_mfr_count: int = 3
    min_mfr_best_prob: float = 0.75
    min_mfr_change_rate: float = 0.50
    max_mfr_entropy: float = 1.25
    long_output_ratio: float = 4.0
    max_free_output_chars: int = 40


def iter_lang_rows(rows: Iterable[Dict[str, Any]], lang: str = "jp") -> Iterable[Tuple[List[str], List[str]]]:
    for row in rows:
        if row.get("lang") != lang:
            continue
        raw = as_list(row.get("raw"))
        norm = normalize_empty_norm(raw, row.get("norm"))
        if len(raw) != len(norm):
            continue
        yield raw, norm


def build_token_stats(rows: Iterable[Dict[str, Any]], lang: str = "jp") -> Dict[str, Any]:
    """Build raw->norm counts and changed vocab from labeled rows."""
    norm_counts: Dict[str, Counter] = defaultdict(Counter)
    total_tokens = 0
    changed_tokens = 0

    for raw_words, norm_words in iter_lang_rows(rows, lang=lang):
        for raw, norm in zip(raw_words, norm_words):
            if norm == "":
                # Empty string is a valid deletion label in train/validation.
                pass
            norm_counts[raw][norm] += 1
            total_tokens += 1
            if raw != norm:
                changed_tokens += 1

    serial_counts = {raw: dict(cnt) for raw, cnt in norm_counts.items()}
    changed_vocab = sorted([raw for raw, cnt in norm_counts.items() if any(norm != raw for norm in cnt)])
    return {
        "norm_counts": serial_counts,
        "changed_vocab": changed_vocab,
        "total_tokens": total_tokens,
        "changed_tokens": changed_tokens,
        "change_rate": changed_tokens / total_tokens if total_tokens else 0.0,
    }


def build_high_conf_mfr(stats: Dict[str, Any], cfg: ArtifactConfig) -> Dict[str, str]:
    mfr: Dict[str, str] = {}
    for raw, cnt_dict in stats.get("norm_counts", {}).items():
        counter = Counter(cnt_dict)
        total = sum(counter.values())
        if total < cfg.min_mfr_count:
            continue
        best_norm, best_count = max(counter.items(), key=lambda kv: (kv[1], kv[0] == raw, kv[0]))
        if best_norm == raw:
            continue
        if raw in MFR_EXCLUDE_TOKENS:
            continue
        best_prob = best_count / total
        change_rate = 1.0 - (counter.get(raw, 0) / total)
        ent = token_entropy(counter)
        if best_prob >= cfg.min_mfr_best_prob and change_rate >= cfg.min_mfr_change_rate and ent <= cfg.max_mfr_entropy:
            mfr[raw] = best_norm
    return mfr


def build_general_mfr(rows: Iterable[Dict[str, Any]], lang: Optional[str] = None) -> Dict[str, str]:
    """Plain MFR for non-Japanese languages."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if lang is not None and row.get("lang") != lang:
            continue
        raw_words = as_list(row.get("raw"))
        norm_words = normalize_empty_norm(raw_words, row.get("norm"))
        if len(raw_words) != len(norm_words):
            continue
        for raw, norm in zip(raw_words, norm_words):
            counts[raw][norm if norm is not None else raw] += 1
    return {raw: max(cnt.items(), key=lambda kv: (kv[1], kv[0] == raw, kv[0]))[0] for raw, cnt in counts.items()}


def build_artifacts(rows: Iterable[Dict[str, Any]], cfg: Optional[ArtifactConfig] = None, lang: str = "jp") -> Dict[str, Any]:
    cfg = cfg or ArtifactConfig()
    stats = build_token_stats(rows, lang=lang)
    high_conf_mfr = build_high_conf_mfr(stats, cfg)

    long_outputs = set()
    for raw, norm in high_conf_mfr.items():
        if raw and len(norm.replace(" ", "")) >= max(8, cfg.long_output_ratio * max(1, len(raw))):
            long_outputs.add(raw)

    return {
        "version": "jp-copy-mfr-contextbyt5-v1",
        "lang": lang,
        "delete_token": DELETE_TOKEN,
        "config": cfg.__dict__,
        "stats_summary": {
            "total_tokens": stats["total_tokens"],
            "changed_tokens": stats["changed_tokens"],
            "change_rate": stats["change_rate"],
            "changed_vocab_size": len(stats["changed_vocab"]),
            "high_conf_mfr_size": len(high_conf_mfr),
        },
        "changed_vocab": stats["changed_vocab"],
        "high_conf_mfr": high_conf_mfr,
        "long_output_allow_raw": sorted(long_outputs),
        # Keep top targets for debugging without storing a huge nested table.
        "top_norms": {
            raw: Counter(cnt).most_common(5)
            for raw, cnt in stats.get("norm_counts", {}).items()
            if raw in stats["changed_vocab"]
        },
    }


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def should_call_byt5(raw: str, words: Sequence[str], i: int, artifacts: Dict[str, Any]) -> bool:
    changed_vocab = set(artifacts.get("changed_vocab", []))
    if raw in CONTEXT_SENSITIVE_TOKENS:
        return True
    if raw in changed_vocab:
        return True
    if raw in PUNCT_END_CANDIDATES and is_near_sentence_end(words, i):
        return True
    if has_japanese(raw) and ("ー" in raw or "〜" in raw or "っ" in raw[-1:]):
        return True
    return False


def safety_filter(raw: str, pred: str, words: Sequence[str], i: int, artifacts: Dict[str, Any]) -> str:
    """Reject suspicious ByT5 outputs and fall back to raw copy."""
    pred = decode_target(pred)
    pred = pred.strip()

    if pred == raw:
        return pred

    # Empty output/deletion is only allowed for known noisy candidates.
    if pred == "":
        if raw in DELETE_CANDIDATES:
            return pred
        # If train's high-confidence MFR explicitly learned deletion, allow it.
        if artifacts.get("high_conf_mfr", {}).get(raw) == "":
            return pred
        return raw

    cfg_dict = artifacts.get("config", {})
    long_ratio = float(cfg_dict.get("long_output_ratio", 4.0))
    max_chars = int(cfg_dict.get("max_free_output_chars", 40))
    compact_raw = raw.replace(" ", "")
    compact_pred = pred.replace(" ", "")

    # Allow long expansions only if this raw token was supported by training MFR.
    if len(compact_pred) > max_chars:
        if raw not in set(artifacts.get("long_output_allow_raw", [])):
            return raw
    if compact_raw and len(compact_pred) > max(12, int(long_ratio * len(compact_raw))):
        if raw not in set(artifacts.get("long_output_allow_raw", [])):
            return raw

    # Avoid bizarre script jumps unless the raw token was observed as changeable.
    changed_vocab = set(artifacts.get("changed_vocab", []))
    if raw not in changed_vocab and raw not in CONTEXT_SENSITIVE_TOKENS:
        raw_sig = script_signature(raw)
        pred_sig = script_signature(pred)
        if raw_sig != pred_sig and pred_sig != "other":
            return raw

    # Hard-protected tokens should not be changed by ByT5.
    if is_hard_keep(raw):
        # Exception: high-confidence MFR may have handled RT/TL/DM/GW before this point;
        # ByT5 should not freely alter protected tokens.
        return raw

    return pred


def first_pass_prediction(raw: str, words: Sequence[str], i: int, artifacts: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Return (prediction, needs_byt5). The caller may batch ByT5 only when needed.
    """
    high_conf_mfr = artifacts.get("high_conf_mfr", {})

    # MFR shortcut comes before hard keep so RT/TL/DM/GW-style abbreviations can be expanded.
    if raw in high_conf_mfr:
        return high_conf_mfr[raw], False

    if is_hard_keep(raw):
        return raw, False

    if should_call_byt5(raw, words, i, artifacts):
        return raw, True

    return raw, False


def sample_training_examples(
    rows: Iterable[Dict[str, Any]],
    artifacts: Dict[str, Any],
    lang: str = "jp",
    unchanged_sample_rate: float = 0.08,
    punct_unchanged_rate: float = 0.50,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Create token-level ByT5 examples biased toward changed/contextual tokens."""
    rng = random.Random(seed)
    changed_vocab = set(artifacts.get("changed_vocab", []))
    examples: List[Dict[str, str]] = []

    for raw_words, norm_words in iter_lang_rows(rows, lang=lang):
        for i, (raw, norm) in enumerate(zip(raw_words, norm_words)):
            changed = raw != norm
            include = False
            reason = "ordinary"

            if changed:
                include = True
                reason = "changed"
            elif raw in CONTEXT_SENSITIVE_TOKENS or raw in changed_vocab:
                include = True
                reason = "unchanged_context_or_seen_changed"
            elif raw in PUNCT_END_CANDIDATES and is_near_sentence_end(raw_words, i):
                include = rng.random() < punct_unchanged_rate
                reason = "unchanged_punct_end"
            else:
                include = rng.random() < unchanged_sample_rate

            if not include:
                continue

            examples.append({
                "input_text": make_byt5_input(lang, raw_words, i),
                "target_text": encode_target(norm),
                "raw": raw,
                "norm": norm,
                "reason": reason,
            })

    return examples


def evaluate_predictions(rows: Iterable[Dict[str, Any]], pred_rows: Iterable[Dict[str, Any]], lang: str = "jp") -> Dict[str, float]:
    gold_tokens = 0
    changed_gold = 0
    correct = 0
    pred_changed = 0
    rows_list = list(rows)
    pred_list = list(pred_rows)
    for gold_row, pred_row in zip(rows_list, pred_list):
        if gold_row.get("lang") != lang:
            continue
        raw = as_list(gold_row.get("raw"))
        gold = normalize_empty_norm(raw, gold_row.get("norm"))
        pred = as_list(pred_row.get("pred"))
        if len(raw) != len(gold) or len(raw) != len(pred):
            continue
        for r, g, p in zip(raw, gold, pred):
            gold_tokens += 1
            if r != g:
                changed_gold += 1
            if r != p:
                pred_changed += 1
            if g == p:
                correct += 1
    acc = correct / gold_tokens if gold_tokens else 0.0
    lai = (gold_tokens - changed_gold) / gold_tokens if gold_tokens else 0.0
    err = (acc - lai) / (1 - lai) if gold_tokens and lai < 1 else 0.0
    return {
        "tokens": gold_tokens,
        "gold_changed": changed_gold,
        "pred_changed": pred_changed,
        "accuracy": acc,
        "lai": lai,
        "err": err,
    }
