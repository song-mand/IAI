#!/usr/bin/env python3
import argparse
import gc
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Optional

import joblib
import numpy as np
import torch
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model


LANG = "ja"


# -----------------------------
# Basic utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_of(raw: str, norm: Optional[str]) -> str:
    return norm if norm is not None else raw


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def is_hiragana(ch: str) -> bool:
    return "\u3040" <= ch <= "\u309f"


def is_katakana(ch: str) -> bool:
    return "\u30a0" <= ch <= "\u30ff"


def is_kanji(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def collapse_repeats(s: str, max_repeat: int = 2) -> str:
    return re.sub(
        r"(.)\1{" + str(max_repeat) + r",}",
        lambda m: m.group(1) * max_repeat,
        s,
    )


def make_ja_key(s: str) -> str:
    """
    Japanese-oriented surface key:
    - NFKC normalization
    - katakana to hiragana
    - lowercase latin
    - collapse long repeated chars
    """
    s = nfkc(s)
    s = kata_to_hira(s)
    s = s.lower()
    s = collapse_repeats(s, max_repeat=2)
    return s


def is_protected_token(token: str) -> bool:
    t = token.strip()

    if not t:
        return True
    if t.startswith("@") or t.startswith("#"):
        return True
    if t.startswith("http://") or t.startswith("https://"):
        return True
    if re.fullmatch(r"\d+([.,:/-]\d+)*", t):
        return True
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True

    return False


def token_shape_ja(token: str) -> str:
    out = []

    for ch in token:
        if is_hiragana(ch):
            out.append("H")
        elif is_katakana(ch):
            out.append("K")
        elif is_kanji(ch):
            out.append("C")
        elif ch.isdigit():
            out.append("0")
        elif ch.isalpha():
            out.append("a")
        else:
            out.append(ch)

    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(out))


def has_long_repetition(token: str) -> bool:
    return re.search(r"(.)\1{2,}", token) is not None


def has_kana(token: str) -> bool:
    return any(is_hiragana(ch) or is_katakana(ch) for ch in token)


def has_kanji(token: str) -> bool:
    return any(is_kanji(ch) for ch in token)


def has_hiragana(token: str) -> bool:
    return any(is_hiragana(ch) for ch in token)


def has_katakana(token: str) -> bool:
    return any(is_katakana(ch) for ch in token)


# -----------------------------
# Dataset loading / split
# -----------------------------

def load_rows(train_file: str, dataset_name: str):
    if train_file and os.path.exists(train_file):
        print(f"Loading local train file: {train_file}")
        return load_dataset("parquet", data_files={"train": train_file})["train"]

    print(f"Loading HF dataset: {dataset_name}")
    return load_dataset(dataset_name, split="train")


def split_lang_rows(rows, lang: str, valid_ratio: float, seed: int):
    lang_rows = [row for row in rows if row["lang"] == lang]
    rng = random.Random(seed)
    rng.shuffle(lang_rows)

    if valid_ratio <= 0:
        return lang_rows, []

    n_valid = max(1, int(len(lang_rows) * valid_ratio))
    valid_rows = lang_rows[:n_valid]
    train_rows = lang_rows[n_valid:]

    return train_rows, valid_rows


# -----------------------------
# MFR
# -----------------------------

def build_mfr(rows):
    counts = defaultdict(Counter)

    for row in rows:
        if row["lang"] != LANG:
            continue

        for raw, norm in zip(row["raw"], row["norm"]):
            target = target_of(raw, norm)
            counts[raw][target] += 1

    mfr = {}
    conf = {}

    for raw, counter in counts.items():
        total = sum(counter.values())
        best, best_count = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
        mfr[raw] = best
        conf[raw] = best_count / total if total else 0.0

    return mfr, conf


# -----------------------------
# JA ByT5 dataset
# -----------------------------

class JaByT5Dataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_length: int = 128,
        unchanged_keep_prob: float = 0.5,
        seed: int = 5,
        changed_repeat: int = 3,
        prompt_format: str = "sentinel",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        rng = random.Random(seed)

        for row in rows:
            if row["lang"] != LANG:
                continue

            raw_words = list(row["raw"])
            norm_words = list(row["norm"])
            context = " ".join(raw_words)

            for i, raw_word in enumerate(raw_words):
                target_word = target_of(raw_word, norm_words[i])
                changed = raw_word != target_word

                if not changed and rng.random() > unchanged_keep_prob:
                    continue

                if prompt_format == "natural":
                    input_text = f"lang: ja word: {raw_word} context: {context}"
                elif prompt_format == "marked_natural":
                    words_copy = list(raw_words)
                    words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                    marked_context = " ".join(words_copy)
                    input_text = f"normalize lang: ja target: {raw_word} context: {marked_context}"
                else:
                    words_copy = list(raw_words)
                    words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                    input_text = " ".join(words_copy)

                repeat = changed_repeat if changed else 1

                for _ in range(repeat):
                    self.samples.append(
                        {
                            "input_text": input_text,
                            "target_text": target_word,
                        }
                    )

        print(
            f"[JA] ByT5 samples: {len(self.samples)} "
            f"(unchanged_keep_prob={unchanged_keep_prob}, changed_repeat={changed_repeat})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        model_inputs = self.tokenizer(
            sample["input_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            sample["target_text"],
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


# -----------------------------
# Detector features
# -----------------------------

def build_detector_stats(rows):
    raw_counts = defaultdict(Counter)
    key_counts = defaultdict(Counter)

    for row in rows:
        if row["lang"] != LANG:
            continue

        for raw, norm in zip(row["raw"], row["norm"]):
            target = target_of(raw, norm)
            raw_counts[raw][target] += 1
            key_counts[make_ja_key(raw)][target] += 1

    raw_stats = {}
    for raw, counter in raw_counts.items():
        total = sum(counter.values())
        copy = counter.get(raw, 0)
        changed = total - copy
        best_norm, best_count = counter.most_common(1)[0]

        raw_stats[raw] = {
            "total": total,
            "copy": copy,
            "changed": changed,
            "change_prob": changed / total if total else 0.0,
            "copy_prob": copy / total if total else 0.0,
            "best_norm": best_norm,
            "best_count": best_count,
            "best_prob": best_count / total if total else 0.0,
        }

    key_stats = {}
    for key, counter in key_counts.items():
        total = sum(counter.values())
        best_norm, best_count = counter.most_common(1)[0]

        key_stats[key] = {
            "total": total,
            "best_norm": best_norm,
            "best_count": best_count,
            "best_prob": best_count / total if total else 0.0,
        }

    return raw_stats, key_stats


def stat_features(raw, raw_stats, key_stats):
    feats = {}

    rs = raw_stats.get(raw)
    if rs is None:
        feats.update(
            {
                "raw_seen": 0,
                "raw_total": 0,
                "raw_change_prob": 0.0,
                "raw_copy_prob": 0.0,
                "raw_best_is_copy": 0,
                "raw_best_prob": 0.0,
            }
        )
    else:
        feats.update(
            {
                "raw_seen": 1,
                "raw_total": min(rs["total"], 10),
                "raw_change_prob": rs["change_prob"],
                "raw_copy_prob": rs["copy_prob"],
                "raw_best_is_copy": int(rs["best_norm"] == raw),
                "raw_best_prob": rs["best_prob"],
            }
        )

    key = make_ja_key(raw)
    ks = key_stats.get(key)

    if ks is None:
        feats.update(
            {
                "key_seen": 0,
                "key_total": 0,
                "key_best_prob": 0.0,
                "key_best_is_raw": 0,
            }
        )
    else:
        feats.update(
            {
                "key_seen": 1,
                "key_total": min(ks["total"], 10),
                "key_best_prob": ks["best_prob"],
                "key_best_is_raw": int(ks["best_norm"] == raw),
            }
        )

    return feats


def detector_features(raw, left, right, raw_stats, key_stats):
    n = nfkc(raw)
    key = make_ja_key(raw)

    chars = len(raw)
    hira = sum(is_hiragana(ch) for ch in raw)
    kata = sum(is_katakana(ch) for ch in raw)
    kanji = sum(is_kanji(ch) for ch in raw)
    digits = sum(ch.isdigit() for ch in raw)
    latin = sum(ch.isascii() and ch.isalpha() for ch in raw)
    punct = sum(not ch.isalnum() for ch in raw)

    feats = {
        "bias": 1,

        "raw=" + raw: 1,
        "nfkc=" + n: 1,
        "key=" + key: 1,
        "shape=" + token_shape_ja(raw): 1,

        "left_key=" + make_ja_key(left): 1,
        "right_key=" + make_ja_key(right): 1,

        "prefix1=" + key[:1]: 1,
        "prefix2=" + key[:2]: 1,
        "prefix3=" + key[:3]: 1,
        "suffix1=" + key[-1:]: 1,
        "suffix2=" + key[-2:]: 1,
        "suffix3=" + key[-3:]: 1,

        "len": min(chars, 30),
        "hira": min(hira, 30),
        "kata": min(kata, 30),
        "kanji": min(kanji, 30),
        "digits": min(digits, 30),
        "latin": min(latin, 30),
        "punct": min(punct, 30),

        "is_protected": int(is_protected_token(raw)),
        "has_hiragana": int(has_hiragana(raw)),
        "has_katakana": int(has_katakana(raw)),
        "has_kanji": int(has_kanji(raw)),
        "has_kana": int(has_kana(raw)),
        "has_long_repetition": int(has_long_repetition(raw)),
        "nfkc_changed": int(raw != n),
        "kata_to_hira_changed": int(kata_to_hira(n) != n),
        "all_katakana": int(kata > 0 and hira == 0 and kanji == 0),
        "all_hiragana": int(hira > 0 and kata == 0 and kanji == 0),
        "mixed_script": int(sum(x > 0 for x in [hira, kata, kanji, latin]) >= 2),
    }

    feats.update(stat_features(raw, raw_stats, key_stats))
    return feats


def build_detector_examples(rows, raw_stats, key_stats):
    X = []
    y = []

    for row in rows:
        if row["lang"] != LANG:
            continue

        raw_words = list(row["raw"])
        norm_words = list(row["norm"])

        for i, raw in enumerate(raw_words):
            norm = target_of(raw, norm_words[i])
            label = int(raw != norm)

            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

            X.append(detector_features(raw, left, right, raw_stats, key_stats))
            y.append(label)

    return X, np.array(y, dtype=np.int64)


# -----------------------------
# Training
# -----------------------------

def choose_base_model_name(preferred: str):
    """
    For Japanese, language-specific ufal model may or may not exist.
    Try preferred first, then fallback to google/byt5-small.
    """
    candidates = [
        preferred,
        "ufal/byt5-small-multilexnorm2021-ja",
        "google/byt5-small",
    ]

    for name in candidates:
        try:
            AutoTokenizer.from_pretrained(name)
            AutoModelForSeq2SeqLM.from_pretrained(name)
            print(f"Using base model: {name}")
            return name
        except Exception as e:
            print(f"Failed to load base model: {name}")
            print(e)

    raise RuntimeError("No usable base model found for Japanese.")


def train_byt5(rows, args):
    base_model_name = choose_base_model_name(args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    dataset = JaByT5Dataset(
        rows=rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        unchanged_keep_prob=args.unchanged_keep_prob,
        seed=args.seed,
        changed_repeat=args.changed_repeat,
        prompt_format=args.prompt_format,
    )

    if len(dataset) == 0:
        raise RuntimeError("Empty JA training dataset.")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported() and args.bf16
    use_fp16 = use_cuda and (not use_bf16) and args.fp16

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_dir + "_tmp",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size if use_cuda else min(args.batch_size, 2),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        logging_steps=20,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()

    os.makedirs(args.output_model_dir, exist_ok=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

    print(f"Saved JA ByT5 model to: {args.output_model_dir}")

    del model, trainer, tokenizer, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train_detector(rows, args):
    raw_stats, key_stats = build_detector_stats(rows)
    X, y = build_detector_examples(rows, raw_stats, key_stats)

    print("\n[JA RF change detector]")
    print("examples:", len(y))
    print("COPY examples:", int((y == 0).sum()))
    print("CHANGE examples:", int((y == 1).sum()))
    print("CHANGE rate:", float(y.mean()) if len(y) else 0.0)

    if args.detector_class_weight == "none":
        class_weight = None
    elif args.detector_class_weight == "custom":
        class_weight = {0: 1.0, 1: args.detector_change_weight}
    else:
        class_weight = args.detector_class_weight

    print("class_weight:", class_weight)

    clf = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=args.detector_n_estimators,
                    max_depth=args.detector_max_depth if args.detector_max_depth > 0 else None,
                    min_samples_leaf=args.detector_min_samples_leaf,
                    min_samples_split=args.detector_min_samples_split,
                    max_features=args.detector_max_features,
                    class_weight=class_weight,
                    random_state=args.seed,
                    n_jobs=args.n_jobs,
                ),
            ),
        ]
    )

    clf.fit(X, y)

    pred = clf.predict(X)
    print("\n[Train detector report]")
    print(classification_report(y, pred, target_names=["COPY", "CHANGE"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))

    os.makedirs(os.path.dirname(args.output_detector), exist_ok=True)

    artifact = {
        "model": clf,
        "raw_stats": raw_stats,
        "key_stats": key_stats,
        "lang": LANG,
        "threshold": args.detector_threshold,
        "detector_type": "random_forest",
        "params": vars(args),
    }

    joblib.dump(artifact, args.output_detector)
    print(f"Saved JA detector to: {args.output_detector}")


# -----------------------------
# Optional holdout eval
# -----------------------------

def safe_byt5_output(raw, pred):
    if pred is None:
        return None

    p = pred.strip()
    if not p:
        return None

    bad_markers = ["lang:", "word:", "context:", "<extra_id", "extra_id"]
    if any(m in p.lower() for m in bad_markers):
        return None

    if len(p) > max(20, len(raw) * 3):
        return None

    if " " in p:
        return None

    return p


def build_inputs(raw_words, indices, prompt_format):
    inputs = []

    for i in indices:
        raw = raw_words[i]

        if prompt_format == "natural":
            context = " ".join(raw_words)
            input_text = f"lang: ja word: {raw} context: {context}"
        elif prompt_format == "marked_natural":
            words_copy = list(raw_words)
            words_copy[i] = f"<extra_id_0> {raw} <extra_id_1>"
            marked_context = " ".join(words_copy)
            input_text = f"normalize lang: ja target: {raw} context: {marked_context}"
        else:
            words_copy = list(raw_words)
            words_copy[i] = f"<extra_id_0> {raw} <extra_id_1>"
            input_text = " ".join(words_copy)

        inputs.append(input_text)

    return inputs


def evaluate_predictions(raw_sents, gold_sents, pred_sents):
    total = 0
    raw_correct = 0
    pred_correct = 0
    changed_total = 0
    changed_correct = 0
    unchanged_total = 0
    unchanged_preserve = 0

    for raw_words, gold_words, pred_words in zip(raw_sents, gold_sents, pred_sents):
        for raw, gold, pred in zip(raw_words, gold_words, pred_words):
            total += 1

            if raw == gold:
                raw_correct += 1

            if pred == gold:
                pred_correct += 1

            if raw != gold:
                changed_total += 1
                if pred == gold:
                    changed_correct += 1
            else:
                unchanged_total += 1
                if pred == raw:
                    unchanged_preserve += 1

    baseline = raw_correct / total * 100 if total else 0.0
    acc = pred_correct / total * 100 if total else 0.0
    err = ((acc - baseline) / (100 - baseline) * 100) if baseline < 100 else 0.0
    changed_acc = changed_correct / changed_total * 100 if changed_total else 0.0
    unchanged_pres = unchanged_preserve / unchanged_total * 100 if unchanged_total else 0.0
    over_change = 100 - unchanged_pres

    print(f"Total tokens:              {total}")
    print(f"Changed tokens:            {changed_total}")
    print(f"Changed rate:              {changed_total / total * 100 if total else 0:.2f}%")
    print(f"Baseline acc.(LAI):        {baseline:.2f}")
    print(f"Accuracy:                  {acc:.2f}")
    print(f"ERR:                       {err:.2f}")
    print(f"Changed token accuracy:    {changed_acc:.2f}")
    print(f"Unchanged preservation:    {unchanged_pres:.2f}")
    print(f"Over-change rate:          {over_change:.2f}")


def eval_holdout(train_rows, valid_rows, args):
    print("\n[JA holdout evaluation]")

    mfr, mfr_conf = build_mfr(train_rows)
    detector_artifact = joblib.load(args.output_detector)
    detector = detector_artifact["model"]
    raw_stats = detector_artifact["raw_stats"]
    key_stats = detector_artifact["key_stats"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.output_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.output_model_dir).to(device)
    model.eval()

    raw_sents = []
    gold_sents = []
    pred_sents = []

    counts = Counter()

    for row in valid_rows:
        raw_words = list(row["raw"])
        gold_words = [target_of(r, n) for r, n in zip(row["raw"], row["norm"])]

        pred_words = list(raw_words)
        byt5_indices = []

        for i, raw in enumerate(raw_words):
            if is_protected_token(raw):
                counts["protected_copy"] += 1
                continue

            mp = mfr.get(raw, raw)
            mc = mfr_conf.get(raw, 0.0)

            if mp != raw and mc >= args.mfr_min_conf:
                pred_words[i] = mp
                counts["mfr_change"] += 1
                continue

            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

            feats = detector_features(raw, left, right, raw_stats, key_stats)
            prob = detector.predict_proba([feats])[0][1]

            if prob >= args.detector_threshold:
                byt5_indices.append(i)
                counts["byt5_requested"] += 1
            else:
                counts["detector_copy"] += 1

        if byt5_indices:
            inputs = build_inputs(raw_words, byt5_indices, args.prompt_format)

            batch = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for idx, p in zip(byt5_indices, decoded):
                raw = raw_words[idx]
                p = safe_byt5_output(raw, p)

                if p is None:
                    counts["byt5_invalid_copy"] += 1
                    pred_words[idx] = raw
                elif p != raw:
                    counts["byt5_change"] += 1
                    pred_words[idx] = p
                else:
                    counts["byt5_rejected_copy"] += 1
                    pred_words[idx] = raw

        raw_sents.append(raw_words)
        gold_sents.append(gold_words)
        pred_sents.append(pred_words)

    print("Decision counts:")
    for k, v in counts.most_common():
        print(f"  {k:24s} {v}")

    evaluate_predictions(raw_sents, gold_sents, pred_sents)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, default="./data/train-00000-of-00001.parquet")
    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")

    parser.add_argument("--output_model_dir", type=str, default="./final_model/ja_model")
    parser.add_argument("--output_detector", type=str, default="./detectors/ja_change_detector_rf.joblib")

    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--valid_ratio", type=float, default=0.0)

    parser.add_argument("--base_model", type=str, default="ufal/byt5-small-multilexnorm2021-ja")
    parser.add_argument("--prompt_format", choices=["sentinel", "natural", "marked_natural"], default="sentinel")

    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--unchanged_keep_prob", type=float, default=0.5)
    parser.add_argument("--changed_repeat", type=int, default=3)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--detector_n_estimators", type=int, default=500)
    parser.add_argument("--detector_max_depth", type=int, default=12)
    parser.add_argument("--detector_min_samples_leaf", type=int, default=1)
    parser.add_argument("--detector_min_samples_split", type=int, default=4)
    parser.add_argument("--detector_max_features", type=str, default="sqrt")
    parser.add_argument(
        "--detector_class_weight",
        choices=["none", "custom", "balanced", "balanced_subsample"],
        default="custom",
    )
    parser.add_argument("--detector_change_weight", type=float, default=3.5)
    parser.add_argument("--detector_threshold", type=float, default=0.55)

    parser.add_argument("--mfr_min_conf", type=float, default=0.65)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()
    set_seed(args.seed)

    rows = load_rows(args.train_file, args.dataset_name)
    train_rows, valid_rows = split_lang_rows(rows, LANG, args.valid_ratio, args.seed)

    print(f"[JA] train rows: {len(train_rows)}")
    print(f"[JA] valid rows: {len(valid_rows)}")

    train_byt5(train_rows, args)
    train_detector(train_rows, args)

    if args.valid_ratio > 0 and len(valid_rows) > 0:
        eval_holdout(train_rows, valid_rows, args)


if __name__ == "__main__":
    main()