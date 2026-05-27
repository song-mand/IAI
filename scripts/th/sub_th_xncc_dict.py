# -*- coding: utf-8 -*-
"""XNCC-style extendable dictionary submission for Thai MultiLexNorm.

This is intentionally conservative:
- neural model is not used
- correct_dict is used only as a validity dictionary
- error_dict is used as the only Thai correction source
- each raw -> candidate pair must satisfy strict evidence thresholds

Put this file in iai_code/scripts/th and run with run_th_xncc_dict_sub.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from th_utils import (
    build_mfr_for_any_lang,
    build_resources,
    edit_distance,
    evaluate_token_level,
    has_thai,
    is_protected_token,
    load_pickle,
    save_pickle,
    target_or_raw,
    thai_normalize_token,
)

ALL_LANGS_DEFAULT = "en,da,de,es,hr,it,nl,sl,sr,tr,iden,trde,id,ja,ko,th,vi"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=env("DATASET", "weerayut/multilexnorm2026-dev-pub"))
    p.add_argument("--train-split", default=env("TRAIN_SPLIT", "train"))
    p.add_argument("--eval-split", default=env("EVAL_SPLIT", "test"))
    p.add_argument("--all-langs", default=env("ALL_LANGS", ALL_LANGS_DEFAULT))
    p.add_argument("--lang", default=env("TARGET_LANG", env("LANG", "th")))

    p.add_argument("--resource-path", default=env("TH_RESOURCE_PATH", "models/th/th_resources.pkl"))
    p.add_argument("--submission-dir", default=env("SUBMISSION_DIR", "submission_files"))
    p.add_argument("--zip-path", default=env("ZIP_PATH", "submission.zip"))
    p.add_argument("--debug-path", default=env("DEBUG_PATH", "submission_files/th_xncc_dict_debug.json"))

    # Strict evidence thresholds for raw -> candidate from error_dict.
    p.add_argument("--min-pair-count", type=int, default=int(env("XNCC_MIN_PAIR_COUNT", "3")))
    p.add_argument("--min-pair-conf", type=float, default=float(env("XNCC_MIN_PAIR_CONF", "0.98")))
    p.add_argument("--min-change-rate", type=float, default=float(env("XNCC_MIN_CHANGE_RATE", "0.90")))
    p.add_argument("--min-total-count", type=int, default=int(env("XNCC_MIN_TOTAL_COUNT", "3")))

    # Candidate validity controls.
    p.add_argument("--require-correct-dict", type=int, default=int(env("XNCC_REQUIRE_CORRECT_DICT", "1")))
    p.add_argument("--protect-nonthai", type=int, default=int(env("XNCC_PROTECT_NONTHAI", "1")))
    p.add_argument("--protect-known-correct", type=int, default=int(env("XNCC_PROTECT_KNOWN_CORRECT", "1")))
    p.add_argument("--known-correct-keep-rate", type=float, default=float(env("XNCC_KNOWN_CORRECT_KEEP_RATE", "0.20")))
    p.add_argument("--max-edit-ratio", type=float, default=float(env("XNCC_MAX_EDIT_RATIO", "0.75")))
    p.add_argument("--max-len-ratio", type=float, default=float(env("XNCC_MAX_LEN_RATIO", "1.80")))
    p.add_argument("--max-len-add", type=int, default=int(env("XNCC_MAX_LEN_ADD", "3")))
    p.add_argument("--allow-long-expansion", type=int, default=int(env("XNCC_ALLOW_LONG_EXPANSION", "0")))
    p.add_argument("--allow-abbrev-expansion", type=int, default=int(env("XNCC_ALLOW_ABBREV_EXPANSION", "0")))

    # Optional blacklist. Example: BLOCK_PAIRS='ก่อ=>ก่อน,ค้า=>คะ,เชี่ย=>เหี้ย'
    p.add_argument("--block-pairs", default=env("XNCC_BLOCK_PAIRS", ""))
    p.add_argument("--print-examples", type=int, default=int(env("PRINT_EXAMPLES", "120")))
    p.add_argument("--print-lang-metrics", type=int, default=int(env("PRINT_LANG_METRICS", "1")))
    return p.parse_args()


def parse_block_pairs(s: str) -> set[Tuple[str, str]]:
    out = set()
    for item in (s or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=>" in item:
            a, b = item.split("=>", 1)
        elif ":" in item:
            a, b = item.split(":", 1)
        else:
            continue
        out.add((a.strip(), b.strip()))
    return out


def load_or_build_resources(args, train_rows):
    if os.path.exists(args.resource_path):
        print(f"loaded resources: {args.resource_path}")
        return load_pickle(args.resource_path)
    print(f"[WARN] resources not found; rebuilding: {args.resource_path}")
    rows = [r for r in train_rows if r.get("lang") == args.lang]
    res = build_resources(rows, lang=args.lang)
    save_pickle(res, args.resource_path)
    return res


def pair_stats(resources: Dict[str, Any], raw: str, cand: str) -> Dict[str, float]:
    counts = resources.get("counts", {}).get(raw, {})
    total = float(sum(counts.values()))
    pair_count = float(counts.get(cand, 0))
    keep_count = float(counts.get(raw, 0))
    changed_count = max(0.0, total - keep_count)
    return {
        "total": total,
        "pair_count": pair_count,
        "keep_count": keep_count,
        "pair_conf": pair_count / max(1.0, total),
        "change_rate": changed_count / max(1.0, total),
        "keep_rate": keep_count / max(1.0, total),
    }


def is_abbrev_like(raw: str) -> bool:
    raw = str(raw)
    return raw.endswith(".") or len(raw) <= 4


def pair_allowed(raw: str, cand: str, resources: Dict[str, Any], args, block_pairs: set[Tuple[str, str]]) -> Tuple[bool, str, Dict[str, float]]:
    raw = str(raw)
    cand = str(cand)
    st = pair_stats(resources, raw, cand)

    if not cand or cand == raw:
        return False, "same_or_empty", st
    if (raw, cand) in block_pairs:
        return False, "blocked_pair", st
    if args.protect_nonthai and is_protected_token(raw):
        return False, "protected_nonthai", st
    if has_thai(raw) and not has_thai(cand):
        return False, "lost_thai", st
    if args.require_correct_dict and cand not in resources.get("correct_set", set()):
        return False, "not_in_correct_dict", st

    if st["total"] < args.min_total_count:
        return False, "low_total", st
    if st["pair_count"] < args.min_pair_count:
        return False, "low_pair_count", st
    if st["pair_conf"] < args.min_pair_conf:
        return False, "low_pair_conf", st
    if st["change_rate"] < args.min_change_rate:
        return False, "low_change_rate", st

    # If raw itself is often correct, do not change it. This blocks ambiguous real-word pairs.
    if args.protect_known_correct and raw in resources.get("correct_set", set()) and st["keep_rate"] >= args.known_correct_keep_rate:
        return False, "known_correct_ambiguous", st

    norm_raw = thai_normalize_token(raw)
    norm_cand = thai_normalize_token(cand)
    dist = edit_distance(norm_raw, norm_cand, max_distance=30)
    edit_ratio = dist / max(1, max(len(norm_raw), len(norm_cand)))
    st["edit_distance"] = float(dist)
    st["edit_ratio"] = float(edit_ratio)
    if edit_ratio > args.max_edit_ratio:
        return False, "far_edit", st

    long_expansion = len(cand) > len(raw) * args.max_len_ratio + args.max_len_add
    if long_expansion and not args.allow_long_expansion:
        if not (args.allow_abbrev_expansion and is_abbrev_like(raw)):
            return False, "long_expansion", st

    return True, "ok", st


def choose_xncc_candidate(raw: str, resources: Dict[str, Any], args, block_pairs: set[Tuple[str, str]], reject_counter: Counter):
    err_map = resources.get("error_dict", {}).get(raw, {})
    if not err_map:
        return raw, "keep", 0.0, [], None

    checked = []
    valid = []
    for cand, cnt in sorted(err_map.items(), key=lambda x: (-x[1], x[0])):
        ok, reason, st = pair_allowed(raw, cand, resources, args, block_pairs)
        item = {
            "cand": cand,
            "reason": reason,
            "ok": ok,
            "count": cnt,
            **{k: round(float(v), 6) for k, v in st.items()},
        }
        checked.append(item)
        if not ok:
            reject_counter[reason] += 1
            continue
        # XNCC-style dict score: dictionary evidence only.
        score = 4.0 * st["pair_conf"] + 2.0 * st["change_rate"] + 0.15 * min(st["pair_count"], 20.0)
        valid.append((cand, score, item))

    if not valid:
        return raw, "keep", 0.0, checked[:5], None
    valid.sort(key=lambda x: x[1], reverse=True)
    cand, score, item = valid[0]
    return cand, "xncc_errdict", score, checked[:5], item


def mfr_predict_sentence(raw_words, mfr_dict):
    return [mfr_dict.get(str(w), str(w)) for w in raw_words]


def main():
    args = parse_args()
    os.makedirs(args.submission_dir, exist_ok=True)
    block_pairs = parse_block_pairs(args.block_pairs)

    print("=" * 80)
    print("Thai XNCC-style extendable dictionary submission")
    print("=" * 80)
    print(vars(args))
    if block_pairs:
        print(f"blocked pairs: {sorted(block_pairs)[:20]} ... total={len(block_pairs)}")

    full = load_dataset(args.dataset)
    train_rows = list(full[args.train_split])
    eval_rows = list(full[args.eval_split])
    all_langs = [x.strip() for x in args.all_langs.split(",") if x.strip()]
    resources = load_or_build_resources(args, train_rows)
    print(
        f"TH resources: rows={resources.get('n_rows')} tokens={resources.get('n_tokens')} "
        f"changed={resources.get('n_changed')} error_keys={len(resources.get('error_dict', {}))} "
        f"correct={len(resources.get('correct_set', set()))}"
    )

    mfr_by_lang = {}
    for lang in all_langs:
        if lang == args.lang:
            continue
        mfr_by_lang[lang] = build_mfr_for_any_lang(train_rows, lang)

    all_predictions = []
    stats = {
        "source_counter": Counter(),
        "reject_reasons": Counter(),
        "changed_tokens": 0,
        "candidate_tokens": 0,
        "checked_candidates": 0,
    }
    debug_samples = []
    raw_gold_by_lang = defaultdict(lambda: {"raw": [], "gold": [], "pred": [], "mfr_pred": []})

    for row in tqdm(eval_rows, desc="all languages"):
        lang = row.get("lang")
        raw_words = [str(x) for x in row.get("raw", [])]
        decisions = []
        if lang == args.lang:
            pred_words = []
            for i, raw in enumerate(raw_words):
                pred, src, score, checked, chosen = choose_xncc_candidate(raw, resources, args, block_pairs, stats["reject_reasons"])
                stats["checked_candidates"] += len(checked)
                if checked:
                    stats["candidate_tokens"] += 1
                if pred != raw:
                    stats["changed_tokens"] += 1
                    stats["source_counter"][src] += 1
                    decisions.append({
                        "i": i,
                        "raw": raw,
                        "pred": pred,
                        "source": src,
                        "score": round(float(score), 6),
                        "chosen_stats": chosen,
                        "checked": checked,
                    })
                else:
                    stats["source_counter"]["keep"] += 1
                pred_words.append(pred)
            if decisions and len(debug_samples) < args.print_examples:
                debug_samples.append({"raw": raw_words, "pred": pred_words, "decisions": decisions})
        else:
            pred_words = mfr_predict_sentence(raw_words, mfr_by_lang.get(lang, {}))

        all_predictions.append({"raw": raw_words, "pred": pred_words, "lang": lang})

        if args.print_lang_metrics and row.get("norm") is not None:
            gold = [target_or_raw(r, n) for r, n in zip(raw_words, row.get("norm"))]
            raw_gold_by_lang[lang]["raw"].append(raw_words)
            raw_gold_by_lang[lang]["gold"].append(gold)
            raw_gold_by_lang[lang]["pred"].append(pred_words)
            if lang == args.lang:
                raw_gold_by_lang[lang]["mfr_pred"].append(mfr_predict_sentence(raw_words, resources.get("mfr", {})))

    json_path = os.path.join(args.submission_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")

    debug_obj = {
        "args": vars(args),
        "stats": {
            "source_counter": dict(stats["source_counter"]),
            "reject_reasons": dict(stats["reject_reasons"]),
            "changed_tokens": stats["changed_tokens"],
            "candidate_tokens": stats["candidate_tokens"],
            "checked_candidates": stats["checked_candidates"],
        },
        "samples": debug_samples,
    }
    os.makedirs(os.path.dirname(args.debug_path) or ".", exist_ok=True)
    with open(args.debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_obj, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Submission complete")
    print(f"predictions: {json_path}")
    print(f"zip:         {args.zip_path}")
    print(f"debug:       {args.debug_path}")
    print("-" * 80)
    print("TH XNCC-dict stats")
    print(f"source_counter={dict(stats['source_counter'])}")
    print(f"reject_reasons={dict(stats['reject_reasons'])}")
    print(f"changed_tokens={stats['changed_tokens']}")
    print(f"candidate_tokens={stats['candidate_tokens']}")
    print(f"checked_candidates={stats['checked_candidates']}")

    if args.print_lang_metrics:
        print("-" * 80)
        print("Metrics on eval split only if gold norm is available.")
        for lang in all_langs:
            data = raw_gold_by_lang.get(lang)
            if not data or not data["raw"]:
                continue
            if lang == args.lang and data["mfr_pred"]:
                print(f"[{lang.upper()}] MFR baseline")
                evaluate_token_level(data["raw"], data["gold"], data["mfr_pred"], info=True, prefix="  ")
            print(f"[{lang.upper()}] XNCC-dict pred")
            evaluate_token_level(data["raw"], data["gold"], data["pred"], info=True, prefix="  ")

    print("-" * 80)
    print("Sample TH modifications")
    shown = 0
    for sample in debug_samples:
        for d in sample["decisions"]:
            cs = d.get("chosen_stats") or {}
            print(
                f"  {d['raw']} -> {d['pred']} src={d['source']} "
                f"score={d['score']:.2f} pair_conf={cs.get('pair_conf')} change_rate={cs.get('change_rate')} total={cs.get('total')}"
            )
            shown += 1
            if shown >= min(args.print_examples, 40):
                break
        if shown >= min(args.print_examples, 40):
            break


if __name__ == "__main__":
    main()
