import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_it_change_detector_rf import extract_features, is_protected_token, LANG


def build_mfr_dictionary(train_data):
    counts = defaultdict(Counter)
    lang_train = train_data.filter(lambda x: x["lang"] == LANG)

    for row in lang_train:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1

    mfr = {}
    conf = {}

    for raw, counter in counts.items():
        total = sum(counter.values())
        best, count = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
        mfr[raw] = best
        conf[raw] = count / total if total else 0.0

    return mfr, conf


def build_sentinel_input(raw_words, i):
    target_word = raw_words[i]
    words_copy = list(raw_words)
    words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
    return " ".join(words_copy)


def predict_selected_tokens(raw_words, indices, model, tokenizer, device, max_input_len=128, num_beams=2):
    if not indices:
        return {}

    inputs_list = [build_sentinel_input(raw_words, i) for i in indices]

    inputs = tokenizer(
        inputs_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=num_beams,
        )

    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {idx: pred.strip() for idx, pred in zip(indices, preds)}


def reject_byt5_prediction(raw, pred):
    if pred is None:
        return True

    pred = pred.strip()

    if pred == "":
        return True

    if is_protected_token(raw) and pred != raw:
        return True

    if len(pred) > max(len(raw) * 3, 20):
        return True

    if "http" in pred or "t.co" in pred:
        return True

    if (pred.startswith("@") or pred.startswith("#")) and pred != raw:
        return True

    if " " in pred and "_" not in pred:
        return True

    # IT ByT5 collapse often creates unrelated frequent Italian words.
    # Keep this conservative; MFR/candidate fallback should be preferred later.
    if raw.isalpha() and pred.isalpha():
        if len(raw) >= 4 and len(pred) <= 2:
            return True

    return False


def detector_predict_sentence(raw_words, detector_artifact, threshold):
    model = detector_artifact["model"]
    raw_stats = detector_artifact["raw_stats"]
    key_stats = detector_artifact["key_stats"]
    case_stats = detector_artifact.get("case_stats", {})

    X = []
    for i, raw in enumerate(raw_words):
        left = raw_words[i - 1] if i > 0 else "<BOS>"
        right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
        X.append(extract_features(raw, left, right, raw_stats, key_stats, case_stats))

    prob_change = model.predict_proba(X)[:, 1]
    pred_change = prob_change >= threshold

    return pred_change, prob_change


def evaluate(raw_sents, gold_sents, pred_sents, max_errors=80):
    total = 0
    correct = 0
    changed = 0
    changed_correct = 0
    unchanged = 0
    unchanged_correct = 0
    over_changed = 0
    errors = []

    for raw_words, gold_words, pred_words in zip(raw_sents, gold_sents, pred_sents):
        if len(raw_words) != len(gold_words):
            raise ValueError("raw/gold length mismatch")
        if len(gold_words) != len(pred_words):
            raise ValueError("gold/pred length mismatch")

        for r, g, p in zip(raw_words, gold_words, pred_words):
            total += 1
            is_correct = (p == g)

            if is_correct:
                correct += 1

            if r != g:
                changed += 1
                if is_correct:
                    changed_correct += 1
            else:
                unchanged += 1
                if p == g:
                    unchanged_correct += 1
                if p != r:
                    over_changed += 1

            if p != g and len(errors) < max_errors:
                errors.append((r, g, p))

    lai = (total - changed) / total if total else 0.0
    accuracy = correct / total if total else 0.0
    err = (accuracy - lai) / (1 - lai) if changed else 0.0
    changed_acc = changed_correct / changed if changed else 0.0
    unchanged_acc = unchanged_correct / unchanged if unchanged else 0.0
    over_change_rate = over_changed / unchanged if unchanged else 0.0

    return {
        "total": total,
        "changed": changed,
        "changed_rate": changed / total if total else 0.0,
        "lai": lai,
        "accuracy": accuracy,
        "err": err,
        "changed_acc": changed_acc,
        "unchanged_acc": unchanged_acc,
        "over_change_rate": over_change_rate,
        "errors": errors,
    }


def print_metrics(title, metrics, show_errors=False):
    print(f"\n[{title}]")
    print(f"Total tokens:              {metrics['total']}")
    print(f"Changed tokens:            {metrics['changed']}")
    print(f"Changed rate:              {metrics['changed_rate'] * 100:.2f}%")
    print(f"Baseline acc.(LAI):        {metrics['lai'] * 100:.2f}")
    print(f"Accuracy:                  {metrics['accuracy'] * 100:.2f}")
    print(f"ERR:                       {metrics['err'] * 100:.2f}")
    print(f"Changed token accuracy:    {metrics['changed_acc'] * 100:.2f}")
    print(f"Unchanged preservation:    {metrics['unchanged_acc'] * 100:.2f}")
    print(f"Over-change rate:          {metrics['over_change_rate'] * 100:.2f}")

    if show_errors:
        print("\nError examples: raw -> gold / pred")
        for r, g, p in metrics["errors"]:
            print(f"{r} -> {g} / {p}")


def build_direct_byt5_predictions(raw_sents, model, tokenizer, device, max_input_len, num_beams):
    pred_sents = []
    for raw_words in tqdm(raw_sents, desc="direct ByT5"):
        indices = list(range(len(raw_words)))
        pred_map = predict_selected_tokens(
            raw_words,
            indices,
            model,
            tokenizer,
            device,
            max_input_len=max_input_len,
            num_beams=num_beams,
        )
        pred_sents.append([pred_map[i] for i in indices])
    return pred_sents


def build_hybrid_predictions(
    raw_sents,
    detector_artifact,
    threshold,
    mfr,
    mfr_conf,
    model,
    tokenizer,
    device,
    mode,
    mfr_min_conf,
    byt5_threshold,
    max_input_len,
    num_beams,
):
    pred_sents = []
    decision_counts = Counter()

    for raw_words in tqdm(raw_sents, desc="hybrid it"):
        pred_change, prob_change = detector_predict_sentence(raw_words, detector_artifact, threshold)
        final = list(raw_words)
        byt5_indices = []

        for i, raw in enumerate(raw_words):
            if is_protected_token(raw):
                decision_counts["protected_copy"] += 1
                final[i] = raw
                continue

            if not pred_change[i]:
                decision_counts["detector_copy"] += 1
                final[i] = raw
                continue

            cand = mfr.get(raw, raw)
            conf = mfr_conf.get(raw, 0.0)

            if mode == "detector_mfr_only":
                if cand != raw:
                    final[i] = cand
                    decision_counts["mfr_change"] += 1
                else:
                    final[i] = raw
                    decision_counts["mfr_no_candidate_copy"] += 1
                continue

            if mode == "detector_mfr_byt5":
                if cand != raw and conf >= mfr_min_conf:
                    final[i] = cand
                    decision_counts["mfr_change"] += 1
                elif prob_change[i] >= byt5_threshold:
                    byt5_indices.append(i)
                    decision_counts["byt5_requested"] += 1
                else:
                    final[i] = raw
                    decision_counts["low_prob_copy"] += 1
                continue

            if mode == "detector_byt5":
                if prob_change[i] >= byt5_threshold:
                    byt5_indices.append(i)
                    decision_counts["byt5_requested"] += 1
                else:
                    final[i] = raw
                    decision_counts["low_prob_copy"] += 1
                continue

            raise ValueError(f"Unknown mode: {mode}")

        if byt5_indices:
            pred_map = predict_selected_tokens(
                raw_words,
                byt5_indices,
                model,
                tokenizer,
                device,
                max_input_len=max_input_len,
                num_beams=num_beams,
            )

            for i in byt5_indices:
                raw = raw_words[i]
                pred = pred_map.get(i, raw)

                if reject_byt5_prediction(raw, pred):
                    cand = mfr.get(raw, raw)
                    if cand != raw:
                        final[i] = cand
                        decision_counts["byt5_rejected_mfr"] += 1
                    else:
                        final[i] = raw
                        decision_counts["byt5_rejected_copy"] += 1
                else:
                    final[i] = pred
                    decision_counts["byt5_change"] += 1

        pred_sents.append(final)

    print("\nHybrid decision counts:")
    for k, v in decision_counts.most_common():
        print(f"  {k:24s} {v}")

    return pred_sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./eval_splits_it/it_train.parquet")
    parser.add_argument("--valid_file", type=str, default="./eval_splits_it/it_valid.parquet")
    parser.add_argument("--detector", type=str, default="./detectors/it_change_detector_rf.joblib")
    parser.add_argument("--model_dir", type=str, default="./final_model_eval_it")
    parser.add_argument("--threshold", type=float, default=0.47)
    parser.add_argument("--byt5_threshold", type=float, default=0.65)
    parser.add_argument("--mode", choices=["detector_mfr_byt5", "detector_byt5", "detector_mfr_only"], default="detector_mfr_byt5")
    parser.add_argument("--mfr_min_conf", type=float, default=0.65)
    parser.add_argument("--max_input_len", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--compare_direct_byt5", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
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

    mfr, mfr_conf = build_mfr_dictionary(train_data)
    mfr_preds = [[mfr.get(w, w) for w in raw_words] for raw_words in raw_sents]
    mfr_metrics = evaluate(raw_sents, gold_sents, mfr_preds)
    print_metrics("MFR baseline", mfr_metrics, show_errors=False)

    detector_artifact = joblib.load(args.detector)

    model_path = os.path.join(args.model_dir, "it_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ByT5 model not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    if args.compare_direct_byt5:
        direct_preds = build_direct_byt5_predictions(
            raw_sents,
            model,
            tokenizer,
            device,
            max_input_len=args.max_input_len,
            num_beams=args.num_beams,
        )
        direct_metrics = evaluate(raw_sents, gold_sents, direct_preds)
        print_metrics("Direct ByT5 baseline", direct_metrics, show_errors=False)

    hybrid_preds = build_hybrid_predictions(
        raw_sents=raw_sents,
        detector_artifact=detector_artifact,
        threshold=args.threshold,
        mfr=mfr,
        mfr_conf=mfr_conf,
        model=model,
        tokenizer=tokenizer,
        device=device,
        mode=args.mode,
        mfr_min_conf=args.mfr_min_conf,
        byt5_threshold=args.byt5_threshold,
        max_input_len=args.max_input_len,
        num_beams=args.num_beams,
    )

    hybrid_metrics = evaluate(raw_sents, gold_sents, hybrid_preds)
    print_metrics(f"Hybrid: {args.mode}", hybrid_metrics, show_errors=args.verbose)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
