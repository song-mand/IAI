import argparse
import os
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


LANG = "es"
COPY_LABEL = "COPY"
NORM_PREFIX = "NORM"


def build_mfr_dictionary(train_data):
    counts = defaultdict(Counter)
    es_train = train_data.filter(lambda x: x["lang"] == LANG)

    for row in es_train:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1

    return {
        raw: max(targets.items(), key=lambda x: (x[1], x[0] == raw))[0]
        for raw, targets in counts.items()
    }


# Same automatic candidate dictionary used by train_es_action_holdout.py.
# No hand-written ES rules are used.
def strip_accents(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_apostrophe(s: str) -> str:
    return (
        s.replace("’", "'")
         .replace("`", "'")
         .replace("´", "'")
    )


def collapse_repeats(s: str, max_repeat: int = 2) -> str:
    import re
    return re.sub(
        r"(.)\1{" + str(max_repeat) + r",}",
        lambda m: m.group(1) * max_repeat,
        s,
    )


def make_key(s: str) -> str:
    s = normalize_apostrophe(s)
    s = s.lower()
    s = strip_accents(s)
    s = collapse_repeats(s, max_repeat=2)
    return s


def build_observed_candidate_dictionary(train_data):
    exact = defaultdict(Counter)
    key_index = defaultdict(Counter)

    es_train = train_data.filter(lambda x: x["lang"] == LANG)

    for row in es_train:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            exact[raw][target] += 1
            key_index[make_key(raw)][target] += 1

    return {
        "exact": exact,
        "key": key_index,
    }


def get_candidates(raw: str, cand_dict, top_k: int = 5):
    if cand_dict is None:
        return []

    merged = Counter()

    if raw in cand_dict["exact"]:
        merged.update(cand_dict["exact"][raw])

    key = make_key(raw)
    if key in cand_dict["key"]:
        merged.update(cand_dict["key"][key])

    return [norm for norm, _ in merged.most_common(top_k) if norm != raw]


def build_action_input(raw_words, i, use_candidates_in_prompt=False, candidate_dict=None):
    raw_word = raw_words[i]
    words_copy = list(raw_words)
    words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
    marked_sentence = " ".join(words_copy)

    input_text = (
        f"task: lexical_normalization "
        f"lang: {LANG} "
        f"target: {raw_word} "
        f"sentence: {marked_sentence}"
    )

    if use_candidates_in_prompt:
        candidates = get_candidates(raw_word, candidate_dict)
        if candidates:
            input_text += " candidates: " + " | ".join(candidates)

    return input_text


def decode_action_output(raw, output):
    if output is None:
        return raw, "MALFORMED"

    out_raw = output.strip()
    out_upper = out_raw.upper()

    # COPY, copy, <COPY> 모두 허용
    if out_upper in {"COPY", "<COPY>"}:
        return raw, "COPY"

    # NORM xxx, norm: xxx, <NORM> xxx 모두 허용
    if out_upper.startswith("NORM "):
        pred = out_raw[5:].strip()
        if pred:
            return pred, "NORM"
        return raw, "MALFORMED"

    if out_upper.startswith("NORM:"):
        pred = out_raw.split(":", 1)[1].strip()
        if pred:
            return pred, "NORM"
        return raw, "MALFORMED"

    if out_upper.startswith("<NORM>"):
        pred = out_raw.replace("<NORM>", "", 1).strip()
        if pred:
            return pred, "NORM"
        return raw, "MALFORMED"

    return raw, "MALFORMED"


def predict_sentence_action(
    raw_words,
    model,
    tokenizer,
    device,
    max_input_len=128,
    num_beams=2,
    use_candidates_in_prompt=False,
    candidate_dict=None,
):
    inputs_list = [
        build_action_input(
            raw_words,
            i,
            use_candidates_in_prompt=use_candidates_in_prompt,
            candidate_dict=candidate_dict,
        )
        for i in range(len(raw_words))
    ]

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

    raw_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    pred_words = []
    actions = []

    for raw, out in zip(raw_words, raw_outputs):
        pred, action = decode_action_output(raw, out)
        pred_words.append(pred)
        actions.append(action)

    return pred_words, raw_outputs, actions


def evaluate(raw_sents, gold_sents, pred_sents, actions_sents=None, max_errors=80):
    total = 0
    correct = 0

    changed = 0
    changed_correct = 0

    unchanged = 0
    unchanged_correct = 0
    over_changed = 0

    errors = []

    action_counter = Counter()
    action_correct = Counter()
    copy_decision_total = 0
    copy_decision_correct = 0
    norm_decision_total = 0
    norm_decision_correct = 0

    for sent_idx, (raw_words, gold_words, pred_words) in enumerate(zip(raw_sents, gold_sents, pred_sents)):
        if len(raw_words) != len(gold_words):
            raise ValueError("raw/gold length mismatch")
        if len(gold_words) != len(pred_words):
            raise ValueError("gold/pred length mismatch")

        actions = actions_sents[sent_idx] if actions_sents is not None else [None] * len(raw_words)

        for r, g, p, action in zip(raw_words, gold_words, pred_words, actions):
            total += 1

            is_changed = (r != g)
            is_correct = (p == g)

            if is_correct:
                correct += 1

            if is_changed:
                changed += 1
                norm_decision_total += 1
                if action == "NORM":
                    norm_decision_correct += 1
                if is_correct:
                    changed_correct += 1
            else:
                unchanged += 1
                copy_decision_total += 1
                if action == "COPY":
                    copy_decision_correct += 1
                if p == g:
                    unchanged_correct += 1
                if p != r:
                    over_changed += 1

            if action is not None:
                action_counter[action] += 1
                if is_correct:
                    action_correct[action] += 1

            if p != g and len(errors) < max_errors:
                errors.append((r, g, p, action))

    lai = (total - changed) / total if total else 0.0
    accuracy = correct / total if total else 0.0
    err = (accuracy - lai) / (1 - lai) if changed else 0.0
    changed_acc = changed_correct / changed if changed else 0.0
    unchanged_acc = unchanged_correct / unchanged if unchanged else 0.0
    over_change_rate = over_changed / unchanged if unchanged else 0.0

    copy_decision_acc = copy_decision_correct / copy_decision_total if copy_decision_total else 0.0
    norm_decision_acc = norm_decision_correct / norm_decision_total if norm_decision_total else 0.0

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
        "copy_decision_acc": copy_decision_acc,
        "norm_decision_acc": norm_decision_acc,
        "action_counter": action_counter,
        "action_correct": action_correct,
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

    if metrics.get("action_counter"):
        print(f"COPY decision accuracy:    {metrics['copy_decision_acc'] * 100:.2f}")
        print(f"NORM decision recall:      {metrics['norm_decision_acc'] * 100:.2f}")
        print("Action counts:")
        for action, count in metrics["action_counter"].most_common():
            correct = metrics["action_correct"].get(action, 0)
            acc = correct / count if count else 0.0
            print(f"  {action:10s} {count:6d}  correct={correct:6d}  acc={acc*100:6.2f}%")

    if show_errors:
        print("\nError examples: raw -> gold / pred / action")
        for r, g, p, action in metrics["errors"]:
            print(f"{r} -> {g} / {p} / {action}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_file", type=str, default="./eval_splits_es/es_valid.parquet")
    parser.add_argument("--train_file", type=str, default="./eval_splits_es/es_train.parquet")
    parser.add_argument("--model_dir", type=str, default="./final_model_eval_es_action")
    parser.add_argument("--max_input_len", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare_mfr", action="store_true")
    parser.add_argument("--use_candidates_in_prompt", action="store_true")
    args = parser.parse_args()

    valid_data = load_dataset(
        "parquet",
        data_files={"valid": args.valid_file},
    )["valid"]
    valid_data = valid_data.filter(lambda x: x["lang"] == LANG)

    train_data = None
    candidate_dict = None

    if args.compare_mfr or args.use_candidates_in_prompt:
        train_data = load_dataset(
            "parquet",
            data_files={"train": args.train_file},
        )["train"]

    if args.use_candidates_in_prompt:
        candidate_dict = build_observed_candidate_dictionary(train_data)
        print("[ES] observed candidate dictionary enabled for eval")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("valid rows:", len(valid_data))

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

    if args.compare_mfr:
        mfr = build_mfr_dictionary(train_data)
        mfr_preds = [
            [mfr.get(w, w) for w in raw_words]
            for raw_words in raw_sents
        ]
        mfr_metrics = evaluate(raw_sents, gold_sents, mfr_preds)
        print_metrics("MFR baseline", mfr_metrics, show_errors=False)

    model_path = os.path.join(args.model_dir, "es_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    pred_sents = []
    raw_output_sents = []
    action_sents = []

    for raw_words in tqdm(raw_sents, desc="eval es action"):
        pred_words, raw_outputs, actions = predict_sentence_action(
            raw_words=raw_words,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_input_len=args.max_input_len,
            num_beams=args.num_beams,
            use_candidates_in_prompt=args.use_candidates_in_prompt,
            candidate_dict=candidate_dict,
        )
        pred_sents.append(pred_words)
        raw_output_sents.append(raw_outputs)
        action_sents.append(actions)

    model_metrics = evaluate(raw_sents, gold_sents, pred_sents, actions_sents=action_sents)
    print_metrics("Action-aware ByT5/LoRA model", model_metrics, show_errors=args.verbose)

    if args.verbose:
        print("\nRaw model output examples:")
        shown = 0
        for raw_words, gold_words, pred_words, raw_outputs, actions in zip(
            raw_sents, gold_sents, pred_sents, raw_output_sents, action_sents
        ):
            for r, g, p, out, action in zip(raw_words, gold_words, pred_words, raw_outputs, actions):
                if shown >= 40:
                    break
                if p != g:
                    print(f"raw={r} gold={g} decoded_pred={p} action={action} raw_output={out}")
                    shown += 1
            if shown >= 40:
                break

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
