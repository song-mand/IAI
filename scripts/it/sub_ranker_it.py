import argparse
import json
import os
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from it_candidate_ranker import ITCandidateRanker


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
IT_LANG = "it"


def safe_norm(raw, norm):
    return raw if norm is None else norm


def get_lang_rows(split, lang: str) -> List[Dict[str, Any]]:
    return [row for row in split if row["lang"] == lang]


def build_mfr_counts(rows: List[Dict[str, Any]]):
    counts = defaultdict(Counter)

    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = safe_norm(raw, norm)
            counts[raw][target] += 1

    return counts


def build_mfr_dict(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    기존 성능 좋았던 MFR 방식:
    가장 빈도 높은 norm 선택.
    동률이면 raw 그대로 유지 선호.
    """
    counts = build_mfr_counts(rows)
    mfr = {}

    for raw, counter in counts.items():
        best = max(
            counter.items(),
            key=lambda x: (x[1], x[0] == raw)
        )[0]
        mfr[raw] = best

    return mfr


def predict_mfr(raw_words: List[str], mfr_dict: Dict[str, str]) -> List[str]:
    return [mfr_dict.get(w, w) for w in raw_words]


def analyze_predictions(lang: str, rows: List[Dict[str, Any]], pred_rows: List[List[str]], max_samples: int = 20):
    total = 0
    changed = 0
    counter = Counter()
    samples = []

    for row, pred in zip(rows, pred_rows):
        raw = row["raw"]
        changes = []

        for r, p in zip(raw, pred):
            total += 1
            if r != p:
                changed += 1
                counter[(r, p)] += 1
                changes.append((r, p))

        if changes and len(samples) < max_samples:
            samples.append({
                "raw": raw,
                "pred": pred,
                "changes": changes,
            })

    return {
        "lang": lang,
        "total_tokens": total,
        "changed_tokens": changed,
        "changed_rate": changed / total if total else 0.0,
        "top_changes": [
            {"raw": r, "pred": p, "count": c}
            for (r, p), c in counter.most_common(100)
        ],
        "samples": samples,
    }


def print_lang_summary(analysis):
    lang = analysis["lang"]

    print(
        f"[{lang.upper():5s}] "
        f"tokens={analysis['total_tokens']:6d} "
        f"changed={analysis['changed_tokens']:5d} "
        f"rate={analysis['changed_rate'] * 100:6.2f}%"
    )


def print_it_detail(analysis):
    print("\n[IT] Prediction analysis")
    print(f"  total tokens:   {analysis['total_tokens']}")
    print(f"  changed tokens: {analysis['changed_tokens']}")
    print(f"  changed rate:   {analysis['changed_rate'] * 100:.2f}%")

    print("\n[IT] Top changes")
    for item in analysis["top_changes"][:80]:
        print(f"  {item['raw']} -> {item['pred']} | count={item['count']}")

    print("\n[IT] Changed samples")
    for i, sample in enumerate(analysis["samples"][:20], start=1):
        print(f"\n  sample {i}")
        print("  raw : " + " ".join(map(str, sample["raw"])))
        print("  pred: " + " ".join(map(str, sample["pred"])))
        print(f"  changes: {sample['changes']}")


def inspect_ranker_decisions(ranker, rows, max_items=120):
    """
    IT ranker가 실제로 어떤 후보 중 무엇을 골랐는지 분석용.
    """
    debug_items = []

    for row_idx, row in enumerate(rows):
        raw_words = [str(x) for x in row["raw"]]

        for i, raw in enumerate(raw_words):
            cands = ranker.generator.candidates(raw, gold=None)

            if len(cands) <= 1:
                continue

            X_dicts = [ranker.features(raw_words, i, cand) for cand in cands]
            X = ranker.vectorizer.transform(X_dicts)
            probs = ranker.model.predict_proba(X)[:, 1]

            cand_probs = sorted(
                [
                    {
                        "candidate": cand,
                        "prob": float(prob),
                        "is_copy": cand == raw,
                    }
                    for cand, prob in zip(cands, probs)
                ],
                key=lambda x: x["prob"],
                reverse=True,
            )

            best = cand_probs[0]
            pred = raw if best["candidate"] != raw and best["prob"] < ranker.threshold else best["candidate"]

            mfr_candidate = ranker.generator.mfr_dict.get(raw, raw)

            item = {
                "row_idx": row_idx,
                "token_idx": i,
                "raw": raw,
                "pred": pred,
                "mfr_candidate": mfr_candidate,
                "threshold": ranker.threshold,
                "candidates": cand_probs,
                "sentence": raw_words,
                "prev": raw_words[i - 1] if i > 0 else "<BOS>",
                "next": raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>",
            }

            if pred != raw:
                item["reason"] = "changed_by_ranker"
                debug_items.append(item)
            elif mfr_candidate != raw:
                item["reason"] = "mfr_rejected_by_ranker"
                debug_items.append(item)
            elif len(cand_probs) >= 2 and abs(cand_probs[0]["prob"] - cand_probs[1]["prob"]) < 0.10:
                item["reason"] = "borderline"
                debug_items.append(item)

    priority = {
        "changed_by_ranker": 0,
        "mfr_rejected_by_ranker": 1,
        "borderline": 2,
    }

    debug_items.sort(
        key=lambda x: (
            priority.get(x["reason"], 9),
            -x["candidates"][0]["prob"],
        )
    )

    return debug_items[:max_items]


def print_it_ranker_debug(debug_items, max_print=60):
    print("\n[IT] Ranker decision debug")

    for idx, item in enumerate(debug_items[:max_print], start=1):
        print(f"\n  debug {idx} | {item['reason']}")
        print(f"  raw={item['raw']} pred={item['pred']} mfr={item['mfr_candidate']}")
        print(f"  prev={item['prev']} next={item['next']}")
        print("  sentence: " + " ".join(map(str, item["sentence"])))
        print("  candidates:")

        for cand in item["candidates"]:
            marker = ""
            if cand["candidate"] == item["pred"]:
                marker = " <-- selected"

            print(
                f"    {cand['candidate']} | prob={cand['prob']:.4f} "
                f"| copy={cand['is_copy']}{marker}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--model", default="scripts/it/ranker_artifacts/it_ranker.pkl")
    parser.add_argument("--out-dir", default="scripts/it/ranker_outputs")
    parser.add_argument("--zip-name", default="submission.zip")
    parser.add_argument("--debug-it", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 50)
    print("1. Load IT ranker")
    print("=" * 50)

    ranker = ITCandidateRanker.load(args.model)

    print(f"model: {args.model}")
    print(f"threshold: {ranker.threshold}")

    print("=" * 50)
    print("2. Load dataset")
    print("=" * 50)

    ds = load_dataset(args.dataset)
    train_split = ds["train"]
    test_split = ds["test"]

    train_langs = sorted(set(row["lang"] for row in train_split))
    test_langs = sorted(set(row["lang"] for row in test_split))

    print(f"train langs: {train_langs}")
    print(f"test langs:  {test_langs}")

    print("=" * 50)
    print("3. Build MFR dictionaries for non-IT languages")
    print("=" * 50)

    mfr_by_lang = {}

    for lang in train_langs:
        if lang == IT_LANG:
            continue

        lang_train = get_lang_rows(train_split, lang)
        mfr_by_lang[lang] = build_mfr_dict(lang_train)

        print(f"[{lang.upper():5s}] MFR entries: {len(mfr_by_lang[lang])}")

    print("=" * 50)
    print("4. Predict all languages")
    print("=" * 50)

    predictions = []
    diagnostics = {
        "method": "IT=ranker, others=MFR",
        "it_model": args.model,
        "it_threshold": ranker.threshold,
        "languages": {},
    }

    for lang in test_langs:
        lang_rows = get_lang_rows(test_split, lang)

        print("\n" + "-" * 50)
        if lang == IT_LANG:
            print(f"[{lang.upper()}] Predict with IT ranker")
        else:
            print(f"[{lang.upper()}] Predict with MFR")
        print("-" * 50)

        pred_rows = []

        for row in tqdm(lang_rows, desc=f"[{lang.upper()}] predict"):
            raw = row["raw"]

            if lang == IT_LANG:
                pred = ranker.predict_sentence(raw)
            else:
                pred = predict_mfr(raw, mfr_by_lang.get(lang, {}))

            if len(raw) != len(pred):
                print(f"[WARN] Length mismatch in {lang}")
                print(f"  raw_len={len(raw)}, pred_len={len(pred)}")
                print("  fallback to raw copy")
                pred = raw.copy()

            pred_rows.append(pred)

            predictions.append({
                "raw": raw,
                "pred": pred,
                "lang": lang,
            })

        analysis = analyze_predictions(lang, lang_rows, pred_rows)
        diagnostics["languages"][lang] = analysis

        print_lang_summary(analysis)

        if lang == IT_LANG:
            print_it_detail(analysis)

            if args.debug_it:
                decision_debug = inspect_ranker_decisions(
                    ranker=ranker,
                    rows=lang_rows,
                    max_items=120,
                )
                diagnostics["it_ranker_decision_debug"] = decision_debug
                print_it_ranker_debug(decision_debug, max_print=60)

    print("=" * 50)
    print("5. Save outputs")
    print("=" * 50)

    pred_json = os.path.join(args.out_dir, "predictions.json")
    diag_json = os.path.join(args.out_dir, "diagnostics.json")
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

    print("=" * 50)
    print("6. Zip check")
    print("=" * 50)

    with zipfile.ZipFile(zip_path, "r") as zipf:
        for info in zipf.infolist():
            print(f"{info.filename} | {info.file_size} bytes")

    print("Done.")


if __name__ == "__main__":
    main()