import argparse
import os
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


STRATEGY = {
    "es": "sentinel",
    "it": "sentinel",
}


def build_mfr_dictionary(train_data, lang):
    counts = defaultdict(Counter)
    lang_data = train_data.filter(lambda x, lang=lang: x["lang"] == lang)

    for row in lang_data:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1

    return {
        raw: max(targets.items(), key=lambda x: (x[1], x[0] == raw))[0]
        for raw, targets in counts.items()
    }


def predict_sentence(raw_words, lang, model, tokenizer, device, fmt, max_input_len=128, num_beams=2):
    inputs_list = []
    context = " ".join(raw_words)

    for i, target_word in enumerate(raw_words):
        if fmt == "natural":
            input_text = f"lang: {lang} word: {target_word} context: {context}"
        else:
            words_copy = list(raw_words)
            words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
            input_text = " ".join(words_copy)

        inputs_list.append(input_text)

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
    return [p.strip() for p in preds]


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

            if p == g:
                correct += 1

            if r != g:
                changed += 1
                if p == g:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_file", type=str, default="./eval_splits/es_it_valid.parquet")
    parser.add_argument("--train_file", type=str, default="./eval_splits/es_it_train.parquet")
    parser.add_argument("--langs", nargs="+", default=["es", "it"])
    parser.add_argument("--model_dir", type=str, default="./final_model_eval")
    parser.add_argument("--max_input_len", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare_mfr", action="store_true")
    args = parser.parse_args()

    valid_data = load_dataset(
        "parquet",
        data_files={"valid": args.valid_file},
    )["valid"]

    train_data = None
    if args.compare_mfr:
        train_data = load_dataset(
            "parquet",
            data_files={"train": args.train_file},
        )["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    for lang in args.langs:
        print(f"\n==================== {lang.upper()} ====================")

        lang_valid = valid_data.filter(lambda x, lang=lang: x["lang"] == lang)
        print("valid rows:", len(lang_valid))

        if len(lang_valid) == 0:
            print(f"No validation rows for lang={lang}")
            continue

        raw_sents = []
        gold_sents = []

        for row in lang_valid:
            raw_words = row["raw"]
            gold_words = [
                n if n is not None else r
                for r, n in zip(row["raw"], row["norm"])
            ]
            raw_sents.append(raw_words)
            gold_sents.append(gold_words)

        if args.compare_mfr:
            mfr = build_mfr_dictionary(train_data, lang)
            mfr_preds = [
                [mfr.get(w, w) for w in raw_words]
                for raw_words in raw_sents
            ]
            mfr_metrics = evaluate(raw_sents, gold_sents, mfr_preds)
            print_metrics("MFR baseline", mfr_metrics, show_errors=False)

        model_path = os.path.join(args.model_dir, f"{lang}_model")
        if not os.path.exists(model_path):
            print("Model not found:", model_path)
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        model.eval()

        fmt = STRATEGY.get(lang, "sentinel")
        pred_sents = []

        for raw_words in tqdm(raw_sents, desc=f"eval {lang}"):
            pred_words = predict_sentence(
                raw_words=raw_words,
                lang=lang,
                model=model,
                tokenizer=tokenizer,
                device=device,
                fmt=fmt,
                max_input_len=args.max_input_len,
                num_beams=args.num_beams,
            )
            pred_sents.append(pred_words)

        model_metrics = evaluate(raw_sents, gold_sents, pred_sents)
        print_metrics("ByT5/LoRA model", model_metrics, show_errors=args.verbose)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
