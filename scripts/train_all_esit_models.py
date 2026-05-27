#!/usr/bin/env python3
"""
Subfinal training script for MultiLexNorm2026 submission.

This file trains on the FULL training data. It does NOT create a validation
split. Use it when final parameters have already been selected and you want to
train the final submission models.

What it trains:
  1. Western languages: same structure as train_western.py
     - natural: en, da, hr, iden, sl, sr
     - sentinel: de, nl, tr, trde
  2. ES / IT ByT5 normalizers
     - sentinel prompt, following train_es_it_holdout.py style
     - ES: epoch 5, unchanged_keep_prob 0.8
     - IT: epoch 2, unchanged_keep_prob 0.8
  3. ES / IT RandomForest COPY/CHANGE detectors
     - trained on the full training data

Default output:
  ./final_model/{lang}_model
  ./detectors/{lang}_change_detector_rf.joblib
"""

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


# -----------------------------------------------------------------------------
# Language settings
# -----------------------------------------------------------------------------

WESTERN_STRATEGY = {
    "da": "natural",
    "en": "natural",
    "sl": "natural",
    "sr": "natural",
    "iden": "natural",
    "hr": "natural",
    "de": "sentinel",
    "nl": "sentinel",
    "tr": "sentinel",
    "trde": "sentinel",
}

ES_IT_STRATEGY = {
    "es": "sentinel",
    "it": "sentinel",
}

# Subfinal setting requested by user.
UNCHANGED_KEEP_PROB = {
    "es": 0.8,
    "it": 0.2,
}

EPOCHS_ES_IT = {
    "es": 5,
    "it": 2,
}

LEARNING_RATE_ES_IT = {
    "es": 1e-5,
    "it": 1e-5,
}

DETECTOR_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_leaf": 1,
    "min_samples_split": 4,
    "max_features": "sqrt",
    "class_weight": "custom",
    "change_weight": 3.0,
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_of(raw: str, norm: Optional[str]) -> str:
    return norm if norm is not None else raw


def load_train_rows(train_file: Optional[str], dataset_name: str):
    if train_file and os.path.exists(train_file):
        print(f"Loading local train file: {train_file}")
        return load_dataset("parquet", data_files={"train": train_file})["train"]

    print(f"Loading HF dataset train split: {dataset_name}")
    return load_dataset(dataset_name, split="train")


# -----------------------------------------------------------------------------
# Seq2Seq datasets
# -----------------------------------------------------------------------------


class WesternByT5Dataset(Dataset):
    def __init__(self, rows, target_lang: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        fmt = WESTERN_STRATEGY.get(target_lang, "natural")

        for row in rows:
            if row["lang"] != target_lang:
                continue

            raw_words = row["raw"]
            norm_words = row["norm"]
            context_sentence = " ".join(raw_words)

            for i, raw_word in enumerate(raw_words):
                target_word = target_of(raw_word, norm_words[i])

                if fmt == "natural":
                    input_text = f"lang: {target_lang} word: {raw_word} context: {context_sentence}"
                else:
                    words_copy = list(raw_words)
                    words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                    input_text = " ".join(words_copy)

                self.samples.append({"input_text": input_text, "target_text": target_word})

        print(f"[{target_lang}] western samples: {len(self.samples)}")

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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


class EsItByT5Dataset(Dataset):
    """ES/IT normalizer dataset.

    This follows the train_es_it_holdout.py style:
      - sentinel prompt
      - keep only a fraction of unchanged tokens
      - repeat changed ES examples to make normalization more active
    """

    def __init__(
        self,
        rows,
        target_lang: str,
        tokenizer,
        max_length: int = 128,
        unchanged_keep_prob: float = 0.8,
        seed: int = 5,
        es_changed_repeat: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        rng = random.Random(seed)
        fmt = ES_IT_STRATEGY.get(target_lang, "sentinel")

        for row in rows:
            if row["lang"] != target_lang:
                continue

            raw_words = row["raw"]
            norm_words = row["norm"]

            for i, raw_word in enumerate(raw_words):
                target_word = target_of(raw_word, norm_words[i])
                changed = raw_word != target_word

                if not changed and rng.random() > unchanged_keep_prob:
                    continue

                if fmt == "natural":
                    context_sentence = " ".join(raw_words)
                    input_text = f"lang: {target_lang} word: {raw_word} context: {context_sentence}"
                else:
                    words_copy = list(raw_words)
                    words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                    input_text = " ".join(words_copy)

                repeat = es_changed_repeat if (changed and target_lang == "es") else 1
                for _ in range(repeat):
                    self.samples.append({"input_text": input_text, "target_text": target_word})

        print(
            f"[{target_lang}] ES/IT samples: {len(self.samples)} "
            f"(unchanged_keep_prob={unchanged_keep_prob})"
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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


# -----------------------------------------------------------------------------
# Seq2Seq training
# -----------------------------------------------------------------------------


def train_seq2seq_model(
    train_dataset: Dataset,
    base_model_name: str,
    output_dir: str,
    epochs: float,
    lr: float,
    batch_size: int,
    max_steps: int,
    use_bf16: bool,
    use_fp16: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
) -> None:
    if len(train_dataset) == 0:
        print(f"Skip empty dataset for {output_dir}")
        return

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    tokenizer = train_dataset.tokenizer

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir + "_tmp",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        logging_steps=20,
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved model to: {output_dir}")

    del model, trainer, tokenizer, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------------------------------
# RF detector
# -----------------------------------------------------------------------------


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_apostrophe(s: str) -> str:
    return s.replace("’", "'").replace("`", "'").replace("´", "'")


def collapse_repeats(s: str, max_repeat: int = 2) -> str:
    return re.sub(r"(.)\1{" + str(max_repeat) + r",}", lambda m: m.group(1) * max_repeat, s)


def make_key(s: str) -> str:
    s = normalize_apostrophe(s).lower()
    s = strip_accents(s)
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


def has_long_repetition(token: str) -> bool:
    return re.search(r"(.)\1{2,}", token.lower()) is not None


def has_laughter_pattern(token: str) -> bool:
    t = token.lower()
    return bool(re.search(r"(ja){2,}", t) or re.search(r"(jaja){1,}", t))


def token_shape(token: str) -> str:
    out = []
    for ch in token:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("0")
        elif strip_accents(ch) != ch:
            out.append("á")
        else:
            out.append(ch)
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(out))


def build_detector_stats(rows, lang: str):
    raw_counts = defaultdict(Counter)
    key_counts = defaultdict(Counter)

    for row in rows:
        if row["lang"] != lang:
            continue
        for raw, norm in zip(row["raw"], row["norm"]):
            target = target_of(raw, norm)
            raw_counts[raw][target] += 1
            key_counts[make_key(raw)][target] += 1

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


def detector_features(raw: str, left: str, right: str, raw_stats, key_stats):
    low = raw.lower()
    key = make_key(raw)
    letters = sum(ch.isalpha() for ch in raw)
    digits = sum(ch.isdigit() for ch in raw)
    punct = sum((not ch.isalnum()) for ch in raw)

    feats = {
        "bias": 1,
        "raw_lower=" + low: 1,
        "key=" + key: 1,
        "shape=" + token_shape(raw): 1,
        "left_lower=" + left.lower(): 1,
        "right_lower=" + right.lower(): 1,
        "prefix1=" + low[:1]: 1,
        "prefix2=" + low[:2]: 1,
        "prefix3=" + low[:3]: 1,
        "suffix1=" + low[-1:]: 1,
        "suffix2=" + low[-2:]: 1,
        "suffix3=" + low[-3:]: 1,
        "len": min(len(raw), 30),
        "letters": min(letters, 30),
        "digits": min(digits, 30),
        "punct": min(punct, 30),
        "is_protected": int(is_protected_token(raw)),
        "starts_at": int(raw.startswith("@")),
        "starts_hash": int(raw.startswith("#")),
        "starts_http": int(raw.startswith("http://") or raw.startswith("https://")),
        "is_digit_like": int(bool(re.fullmatch(r"\d+([.,:/-]\d+)*", raw))),
        "has_long_repetition": int(has_long_repetition(raw)),
        "has_laughter_pattern": int(has_laughter_pattern(raw)),
        "has_accent": int(strip_accents(raw) != raw),
        "is_all_lower": int(raw.islower()),
        "is_all_upper": int(raw.isupper()),
        "has_qkx": int(any(ch in low for ch in ["q", "k", "x"])),
        "has_underscore": int("_" in raw),
    }

    rs = raw_stats.get(raw)
    if rs is None:
        feats.update({
            "raw_seen": 0,
            "raw_total": 0,
            "raw_change_prob": 0.0,
            "raw_copy_prob": 0.0,
            "raw_best_is_copy": 0,
            "raw_best_prob": 0.0,
        })
    else:
        feats.update({
            "raw_seen": 1,
            "raw_total": min(rs["total"], 10),
            "raw_change_prob": rs["change_prob"],
            "raw_copy_prob": rs["copy_prob"],
            "raw_best_is_copy": int(rs["best_norm"] == raw),
            "raw_best_prob": rs["best_prob"],
        })

    ks = key_stats.get(key)
    if ks is None:
        feats.update({"key_seen": 0, "key_total": 0, "key_best_prob": 0.0, "key_best_is_raw": 0})
    else:
        feats.update({
            "key_seen": 1,
            "key_total": min(ks["total"], 10),
            "key_best_prob": ks["best_prob"],
            "key_best_is_raw": int(ks["best_norm"] == raw),
        })
    return feats


def build_detector_examples(rows, lang: str, raw_stats, key_stats):
    X, y = [], []
    for row in rows:
        if row["lang"] != lang:
            continue
        raw_words = row["raw"]
        norm_words = row["norm"]
        for i, raw in enumerate(raw_words):
            norm = target_of(raw, norm_words[i])
            label = int(raw != norm)
            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
            X.append(detector_features(raw, left, right, raw_stats, key_stats))
            y.append(label)
    return X, np.array(y, dtype=np.int64)


def parse_class_weight(mode: str, change_weight: float):
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode == "balanced_subsample":
        return "balanced_subsample"
    if mode == "custom":
        return {0: 1.0, 1: float(change_weight)}
    raise ValueError(f"unknown class_weight mode: {mode}")


def train_change_detector(rows, lang: str, output: str, args) -> None:
    raw_stats, key_stats = build_detector_stats(rows, lang)
    X, y = build_detector_examples(rows, lang, raw_stats, key_stats)

    print(f"\n[{lang.upper()} RF change detector]")
    print("examples:", len(y))
    print("COPY examples:", int((y == 0).sum()))
    print("CHANGE examples:", int((y == 1).sum()))
    print("CHANGE rate:", float(y.mean()) if len(y) else 0.0)

    class_weight = parse_class_weight(args.detector_class_weight, args.detector_change_weight)
    print("class_weight:", class_weight)

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=args.detector_n_estimators,
            max_depth=args.detector_max_depth if args.detector_max_depth > 0 else None,
            min_samples_leaf=args.detector_min_samples_leaf,
            min_samples_split=args.detector_min_samples_split,
            max_features=args.detector_max_features,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )),
    ])

    clf.fit(X, y)

    train_pred = clf.predict(X)
    print("\n[Train detector report: RandomForest]")
    print(classification_report(y, train_pred, target_names=["COPY", "CHANGE"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, train_pred))

    os.makedirs(os.path.dirname(output), exist_ok=True)
    artifact = {
        "model": clf,
        "raw_stats": raw_stats,
        "key_stats": key_stats,
        "lang": lang,
        "threshold": 0.48,
        "detector_type": "random_forest",
        "params": vars(args),
    }
    joblib.dump(artifact, output)
    print("saved:", output)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def cmd_train_subfinal(args):
    set_seed(args.seed)
    rows = load_train_rows(args.train_file, args.dataset_name)

    os.makedirs(args.final_model_dir, exist_ok=True)
    os.makedirs(args.detector_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported() and args.bf16
    use_fp16 = use_cuda and (not use_bf16) and args.fp16

    print("CUDA available:", use_cuda)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
        print("bf16:", use_bf16, "fp16:", use_fp16)

    if args.train_western:
        for lang in [x.strip() for x in args.western_langs.split(",") if x.strip()]:
            fmt = WESTERN_STRATEGY.get(lang, "natural")
            base_model_name = f"ufal/byt5-small-multilexnorm2021-{lang}"
            print(f"\n[{lang.upper()}] western training: format={fmt}, base={base_model_name}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                # Quick load check before building trainer.
                _ = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                del _
            except Exception as e:
                print(f"[{lang.upper()}] skip: failed to load {base_model_name}")
                print(e)
                continue

            dataset = WesternByT5Dataset(rows, lang, tokenizer, max_length=args.max_length)
            if len(dataset) == 0:
                print(f"[{lang.upper()}] skip empty dataset")
                continue

            epochs = args.western_natural_epochs if fmt == "natural" else args.western_sentinel_epochs
            train_seq2seq_model(
                dataset,
                base_model_name,
                os.path.join(args.final_model_dir, f"{lang}_model"),
                epochs=epochs,
                lr=args.western_lr,
                batch_size=args.batch_size if use_cuda else min(args.batch_size, 2),
                max_steps=args.max_steps,
                use_bf16=use_bf16,
                use_fp16=use_fp16,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                seed=args.seed,
            )

    if args.train_es_it:
        for lang in [x.strip() for x in args.es_it_langs.split(",") if x.strip()]:
            if lang not in {"es", "it"}:
                print(f"Skip unsupported es/it lang: {lang}")
                continue

            base_model_name = f"ufal/byt5-small-multilexnorm2021-{lang}"
            print(
                f"\n[{lang.upper()}] subfinal ES/IT training: "
                f"base={base_model_name}, "
                f"epochs={EPOCHS_ES_IT[lang]}, "
                f"unchanged_keep_prob={UNCHANGED_KEEP_PROB[lang]}"
            )

            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            dataset = EsItByT5Dataset(
                rows,
                lang,
                tokenizer,
                max_length=args.max_length,
                unchanged_keep_prob=UNCHANGED_KEEP_PROB[lang],
                seed=args.seed,
                es_changed_repeat=args.es_changed_repeat,
            )

            if len(dataset) == 0:
                print(f"[{lang.upper()}] skip empty dataset")
                continue

            train_seq2seq_model(
                dataset,
                base_model_name,
                os.path.join(args.final_model_dir, f"{lang}_model"),
                epochs=EPOCHS_ES_IT[lang],
                lr=LEARNING_RATE_ES_IT[lang],
                batch_size=args.batch_size if use_cuda else min(args.batch_size, 2),
                max_steps=args.max_steps,
                use_bf16=use_bf16,
                use_fp16=use_fp16,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                seed=args.seed,
            )

            detector_output = os.path.join(args.detector_dir, f"{lang}_change_detector_rf.joblib")
            train_change_detector(rows, lang, detector_output, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./data/train-00000-of-00001.parquet")
    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--final_model_dir", type=str, default="./final_model")
    parser.add_argument("--detector_dir", type=str, default="./detectors")

    # Seed 5 is used because prior ES/IT RF-hybrid experiments were tuned with train_seed=5.
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=-1, help="Set small value, e.g. 50, for smoke test.")

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--train_western", action="store_true", default=True)
    parser.add_argument("--no_train_western", dest="train_western", action="store_false")
    parser.add_argument("--western_langs", type=str, default=",".join(WESTERN_STRATEGY.keys()))
    parser.add_argument("--western_lr", type=float, default=3e-5)
    parser.add_argument("--western_natural_epochs", type=float, default=1)
    parser.add_argument("--western_sentinel_epochs", type=float, default=2)

    parser.add_argument("--train_es_it", action="store_true", default=True)
    parser.add_argument("--no_train_es_it", dest="train_es_it", action="store_false")
    parser.add_argument("--es_it_langs", type=str, default="es,it")
    parser.add_argument("--es_changed_repeat", type=int, default=3)

    parser.add_argument("--detector_n_estimators", type=int, default=DETECTOR_PARAMS["n_estimators"])
    parser.add_argument("--detector_max_depth", type=int, default=DETECTOR_PARAMS["max_depth"])
    parser.add_argument("--detector_min_samples_leaf", type=int, default=DETECTOR_PARAMS["min_samples_leaf"])
    parser.add_argument("--detector_min_samples_split", type=int, default=DETECTOR_PARAMS["min_samples_split"])
    parser.add_argument("--detector_max_features", type=str, default=DETECTOR_PARAMS["max_features"])
    parser.add_argument(
        "--detector_class_weight",
        choices=["none", "custom", "balanced", "balanced_subsample"],
        default=DETECTOR_PARAMS["class_weight"],
    )
    parser.add_argument("--detector_change_weight", type=float, default=DETECTOR_PARAMS["change_weight"])
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()
    cmd_train_subfinal(args)


if __name__ == "__main__":
    main()
