import json
import os
import zipfile
from collections import Counter, defaultdict

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
LANG = "it"

OUT_DIR = "scripts/it/outputs"
PRED_JSON = os.path.join(OUT_DIR, "it_predictions.json")
DIAG_JSON = os.path.join(OUT_DIR, "it_diagnostics.json")
ZIP_PATH = os.path.join(OUT_DIR, "it_submission.zip")


def safe_norm(raw, norm):
    return raw if norm is None else norm


def get_lang_rows(split, lang):
    return [row for row in split if row["lang"] == lang]


def build_mfr_counts(rows):
    counts = defaultdict(Counter)

    for row in rows:
        raw_words = row["raw"]
        norm_words = row["norm"]

        for raw, norm in zip(raw_words, norm_words):
            target = safe_norm(raw, norm)
            counts[raw][target] += 1

    return counts


def build_mfr_dict(mfr_counts):
    """
    기존 성능이 좋았던 방식:
    가장 빈도 높은 norm 선택.
    동률이면 raw 그대로 유지하는 쪽 선호.
    """
    mfr = {}

    for raw, counter in mfr_counts.items():
        best = max(
            counter.items(),
            key=lambda x: (x[1], x[0] == raw)
        )[0]
        mfr[raw] = best

    return mfr


def predict_mfr(raw_words, mfr_dict):
    return [mfr_dict.get(w, w) for w in raw_words]


def summarize_train(rows):
    sent_count = len(rows)
    token_count = 0
    changed_count = 0

    raw_counter = Counter()
    norm_counter = Counter()
    changed_pair_counter = Counter()

    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            norm = safe_norm(raw, norm)

            token_count += 1
            raw_counter[raw] += 1
            norm_counter[norm] += 1

            if raw != norm:
                changed_count += 1
                changed_pair_counter[(raw, norm)] += 1

    return {
        "sentences": sent_count,
        "tokens": token_count,
        "changed_tokens": changed_count,
        "changed_rate": changed_count / token_count if token_count else 0.0,
        "top_raw_tokens": raw_counter.most_common(50),
        "top_norm_tokens": norm_counter.most_common(50),
        "top_gold_changes": [
            {"raw": r, "norm": n, "count": c}
            for (r, n), c in changed_pair_counter.most_common(100)
        ],
    }


def analyze_mfr(mfr_counts, mfr_dict):
    changed_mappings = []
    ambiguous_mappings = []
    identity_count = 0

    for raw, counter in mfr_counts.items():
        chosen = mfr_dict.get(raw, raw)
        total = sum(counter.values())
        chosen_count = counter[chosen]
        conf = chosen_count / total if total else 0.0

        if raw == chosen:
            identity_count += 1
        else:
            changed_mappings.append({
                "raw": raw,
                "chosen": chosen,
                "count": chosen_count,
                "total": total,
                "confidence": conf,
                "candidates": dict(counter.most_common(20)),
            })

        if len(counter) >= 2:
            ambiguous_mappings.append({
                "raw": raw,
                "chosen": chosen,
                "count": chosen_count,
                "total": total,
                "confidence": conf,
                "candidates": dict(counter.most_common(20)),
            })

    changed_mappings.sort(
        key=lambda x: (x["count"], x["confidence"], x["raw"]),
        reverse=True,
    )

    ambiguous_mappings.sort(
        key=lambda x: (len(x["candidates"]), x["total"], x["confidence"]),
        reverse=True,
    )

    return {
        "raw_type_count": len(mfr_counts),
        "identity_mapping_count": identity_count,
        "changed_mapping_count": len(changed_mappings),
        "ambiguous_mapping_count": len(ambiguous_mappings),
        "top_changed_mappings": changed_mappings[:100],
        "top_ambiguous_mappings": ambiguous_mappings[:100],
    }


def analyze_predictions(rows, pred_rows):
    total_tokens = 0
    changed_tokens = 0

    change_counter = Counter()
    changed_samples = []

    for row, pred_words in zip(rows, pred_rows):
        raw_words = row["raw"]
        sent_changes = []

        for raw, pred in zip(raw_words, pred_words):
            total_tokens += 1

            if raw != pred:
                changed_tokens += 1
                change_counter[(raw, pred)] += 1
                sent_changes.append((raw, pred))

        if sent_changes and len(changed_samples) < 50:
            changed_samples.append({
                "raw_sentence": raw_words,
                "pred_sentence": pred_words,
                "changes": sent_changes,
            })

    return {
        "total_tokens": total_tokens,
        "changed_tokens": changed_tokens,
        "changed_rate": changed_tokens / total_tokens if total_tokens else 0.0,
        "top_applied_changes": [
            {"raw": r, "pred": p, "count": c}
            for (r, p), c in change_counter.most_common(100)
        ],
        "changed_samples": changed_samples,
    }


def print_train_summary(summary):
    print("\n[IT] Train summary")
    print(f"  sentences:      {summary['sentences']}")
    print(f"  tokens:         {summary['tokens']}")
    print(f"  changed tokens: {summary['changed_tokens']}")
    print(f"  changed rate:   {summary['changed_rate'] * 100:.2f}%")

    print("\n[IT] Top gold changes in train")
    for item in summary["top_gold_changes"][:50]:
        print(f"  {item['raw']} -> {item['norm']} | count={item['count']}")


def print_mfr_analysis(analysis):
    print("\n[IT] MFR analysis")
    print(f"  raw types:          {analysis['raw_type_count']}")
    print(f"  identity mappings:  {analysis['identity_mapping_count']}")
    print(f"  changed mappings:   {analysis['changed_mapping_count']}")
    print(f"  ambiguous mappings: {analysis['ambiguous_mapping_count']}")

    print("\n[IT] Top changed MFR mappings")
    for item in analysis["top_changed_mappings"][:80]:
        print(
            f"  {item['raw']} -> {item['chosen']} "
            f"| count={item['count']}/{item['total']} "
            f"| conf={item['confidence']:.3f}"
        )

    print("\n[IT] Top ambiguous MFR mappings")
    for item in analysis["top_ambiguous_mappings"][:50]:
        print(
            f"  {item['raw']} | chosen={item['chosen']} "
            f"| count={item['count']}/{item['total']} "
            f"| conf={item['confidence']:.3f} "
            f"| candidates={item['candidates']}"
        )


def print_prediction_analysis(analysis):
    print("\n[IT] Prediction analysis on test")
    print(f"  total tokens:   {analysis['total_tokens']}")
    print(f"  changed tokens: {analysis['changed_tokens']}")
    print(f"  changed rate:   {analysis['changed_rate'] * 100:.2f}%")

    print("\n[IT] Top applied changes on test")
    for item in analysis["top_applied_changes"][:80]:
        print(f"  {item['raw']} -> {item['pred']} | count={item['count']}")

    print("\n[IT] Changed sentence samples")
    for idx, sample in enumerate(analysis["changed_samples"][:20], start=1):
        print(f"\n  sample {idx}")
        print("  raw :  " + " ".join(map(str, sample["raw_sentence"])))
        print("  pred:  " + " ".join(map(str, sample["pred_sentence"])))
        print(f"  changes: {sample['changes']}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 50)
    print("1. Load dataset")
    print("=" * 50)

    dataset = load_dataset(DATASET_NAME)

    train_rows = get_lang_rows(dataset["train"], LANG)
    test_rows = get_lang_rows(dataset["test"], LANG)

    print(f"IT train rows: {len(train_rows)}")
    print(f"IT test rows:  {len(test_rows)}")

    print("=" * 50)
    print("2. Build IT MFR")
    print("=" * 50)

    train_summary = summarize_train(train_rows)
    print_train_summary(train_summary)

    mfr_counts = build_mfr_counts(train_rows)
    mfr_dict = build_mfr_dict(mfr_counts)

    mfr_analysis = analyze_mfr(mfr_counts, mfr_dict)
    print_mfr_analysis(mfr_analysis)

    print("=" * 50)
    print("3. Predict IT test")
    print("=" * 50)

    predictions = []
    pred_rows = []

    for row in tqdm(test_rows, desc="[IT] predict"):
        raw_words = row["raw"]
        pred_words = predict_mfr(raw_words, mfr_dict)

        if len(raw_words) != len(pred_words):
            raise RuntimeError(
                f"Length mismatch: raw={len(raw_words)}, pred={len(pred_words)}"
            )

        pred_rows.append(pred_words)

        predictions.append({
            "raw": raw_words,
            "pred": pred_words,
            "lang": LANG,
        })

    pred_analysis = analyze_predictions(test_rows, pred_rows)
    print_prediction_analysis(pred_analysis)

    diagnostics = {
        "lang": LANG,
        "method": "exact_mfr_with_identity_tie_break",
        "train_summary": train_summary,
        "mfr_analysis": mfr_analysis,
        "prediction_analysis": pred_analysis,
    }

    print("=" * 50)
    print("4. Save outputs")
    print("=" * 50)

    with open(PRED_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    with open(DIAG_JSON, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(PRED_JSON, arcname="predictions.json")

    print(f"Saved predictions: {PRED_JSON}")
    print(f"Saved diagnostics: {DIAG_JSON}")
    print(f"Saved zip:         {ZIP_PATH}")

    print("=" * 50)
    print("5. Zip check")
    print("=" * 50)

    with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
        for info in zipf.infolist():
            print(f"{info.filename} | {info.file_size} bytes")

    print("Done.")


if __name__ == "__main__":
    main()