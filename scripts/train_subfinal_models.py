#!/usr/bin/env python3
"""
Final training / holdout-evaluation script for MultiLexNorm-style submission.

One Python file only:
  1) train_final: train western ByT5 models + ES/IT ByT5 normalizers + ES/IT RF change detectors
  2) eval_es_it: create holdout splits and evaluate ES/IT RF + MFR + ByT5 hybrid

Expected project layout:
  ~/iai_code/
    data/train-00000-of-00001.parquet   # optional; otherwise HF dataset is used
    final_model/{lang}_model/
    detectors/{lang}_change_detector_rf.joblib
"""

import argparse
import gc
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


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

ES_IT_LANGS = ["es", "it"]

UNCHANGED_KEEP_PROB = {"es": 0.8, "it": 0.8}
EPOCHS_ES_IT = {"es": 5, "it": 5}
LEARNING_RATE_ES_IT = {"es": 1e-5, "it": 1e-5}

DEFAULT_DETECTOR_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_leaf": 1,
    "min_samples_split": 4,
    "max_features": "sqrt",
    "class_weight": "custom",
    "change_weight": 3.5,
}

DEFAULT_HYBRID = {
    "es": {"threshold": 0.48, "mfr_min_conf": 0.65},
    "it": {"threshold": 0.55, "mfr_min_conf": 0.65},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_rows(train_file: Optional[str] = None, dataset_name: str = "weerayut/multilexnorm2026-dev-pub"):
    if train_file and os.path.exists(train_file):
        return load_dataset("parquet", data_files={"train": train_file})["train"]
    return load_dataset(dataset_name, split="train")


def target_of(raw: str, norm: Optional[str]) -> str:
    return norm if norm is not None else raw


# -----------------------------------------------------------------------------
# Shared metrics
# -----------------------------------------------------------------------------


def evaluate_predictions(raw_sents: List[List[str]], gold_sents: List[List[str]], pred_sents: List[List[str]]):
    total = 0
    correct = 0
    changed_total = 0
    changed_correct = 0
    unchanged_total = 0
    unchanged_preserved = 0
    errors = []

    for raw_words, gold_words, pred_words in zip(raw_sents, gold_sents, pred_sents):
        for raw, gold, pred in zip(raw_words, gold_words, pred_words):
            total += 1
            if pred == gold:
                correct += 1
            elif len(errors) < 120:
                errors.append((raw, gold, pred))

            if raw != gold:
                changed_total += 1
                if pred == gold:
                    changed_correct += 1
            else:
                unchanged_total += 1
                if pred == raw:
                    unchanged_preserved += 1

    baseline_acc = unchanged_total / total if total else 0.0
    acc = correct / total if total else 0.0
    changed_acc = changed_correct / changed_total if changed_total else 0.0
    unchanged_pres = unchanged_preserved / unchanged_total if unchanged_total else 0.0
    over_change = 1.0 - unchanged_pres if unchanged_total else 0.0
    err = ((acc - baseline_acc) / (1.0 - baseline_acc) * 100.0) if baseline_acc < 1.0 else 0.0

    return {
        "total": total,
        "changed_total": changed_total,
        "changed_rate": changed_total / total if total else 0.0,
        "baseline_acc": baseline_acc,
        "accuracy": acc,
        "err": err,
        "changed_acc": changed_acc,
        "unchanged_preservation": unchanged_pres,
        "over_change": over_change,
        "errors": errors,
    }


def print_metrics(title: str, metrics: dict, show_errors: bool = False):
    print(f"\n[{title}]")
    print(f"Total tokens:              {metrics['total']}")
    print(f"Changed tokens:            {metrics['changed_total']}")
    print(f"Changed rate:              {metrics['changed_rate'] * 100:.2f}%")
    print(f"Baseline acc.(LAI):        {metrics['baseline_acc'] * 100:.2f}")
    print(f"Accuracy:                  {metrics['accuracy'] * 100:.2f}")
    print(f"ERR:                       {metrics['err']:.2f}")
    print(f"Changed token accuracy:    {metrics['changed_acc'] * 100:.2f}")
    print(f"Unchanged preservation:    {metrics['unchanged_preservation'] * 100:.2f}")
    print(f"Over-change rate:          {metrics['over_change'] * 100:.2f}")
    if show_errors:
        print("\nError examples: raw -> gold / pred")
        for r, g, p in metrics["errors"]:
            print(f"{r} -> {g} / {p}")


# -----------------------------------------------------------------------------
# Western ByT5 training
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
        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


class EsItByT5Dataset(Dataset):
    def __init__(
        self,
        rows,
        target_lang: str,
        tokenizer,
        max_length: int = 128,
        unchanged_keep_prob: float = 0.8,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        rng = random.Random(seed)

        for row in rows:
            if row["lang"] != target_lang:
                continue

            raw_words = row["raw"]
            norm_words = row["norm"]
            context_sentence = " ".join(raw_words)

            for i, raw_word in enumerate(raw_words):
                target_word = target_of(raw_word, norm_words[i])
                changed = raw_word != target_word
                if not changed and rng.random() > unchanged_keep_prob:
                    continue

                input_text = f"lang: {target_lang} word: {raw_word} context: {context_sentence}"
                self.samples.append({"input_text": input_text, "target_text": target_word})

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
        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


def train_seq2seq_model(
    train_dataset: Dataset,
    base_model_name: str,
    output_dir: str,
    epochs: float,
    lr: float,
    batch_size: int,
    bf16: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
):
    if len(train_dataset) == 0:
        print(f"skip empty dataset for {output_dir}")
        return

    tokenizer = train_dataset.tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    if get_peft_model is not None:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir + "_tmp",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        bf16=bf16,
        report_to="none",
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    final_model = trainer.model.merge_and_unload() if hasattr(trainer.model, "merge_and_unload") else trainer.model
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, trainer, tokenizer, final_model
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
    raise ValueError(mode)


def train_change_detector(rows, lang: str, output: str, args):
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
        "threshold": 0.5,
        "detector_type": "random_forest",
        "params": vars(args),
    }
    joblib.dump(artifact, output)
    print("saved:", output)


# -----------------------------------------------------------------------------
# MFR / hybrid eval
# -----------------------------------------------------------------------------


def build_mfr(rows, lang: str):
    counts = defaultdict(Counter)
    for row in rows:
        if row["lang"] != lang:
            continue
        for raw, norm in zip(row["raw"], row["norm"]):
            counts[raw][target_of(raw, norm)] += 1

    mfr = {}
    conf = {}
    for raw, counter in counts.items():
        total = sum(counter.values())
        best, best_count = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
        mfr[raw] = best
        conf[raw] = best_count / total if total else 0.0
    return mfr, conf


def detector_prob(detector_artifact, raw_words: List[str], idx: int) -> float:
    raw = raw_words[idx]
    left = raw_words[idx - 1] if idx > 0 else "<BOS>"
    right = raw_words[idx + 1] if idx + 1 < len(raw_words) else "<EOS>"
    feats = detector_features(raw, left, right, detector_artifact["raw_stats"], detector_artifact["key_stats"])
    model = detector_artifact["model"]
    return float(model.predict_proba([feats])[0][1])


def predict_byt5_word(tokenizer, model, device, lang: str, raw_word: str, raw_words: List[str]) -> str:
    context = " ".join(raw_words)
    input_text = f"lang: {lang} word: {raw_word} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=2)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return pred if pred else raw_word


def predict_hybrid(
    rows,
    lang: str,
    detector_path: str,
    model_dir: str,
    threshold: float,
    mfr_min_conf: float,
    show_progress: bool = True,
):
    detector = joblib.load(detector_path)
    mfr, mfr_conf = build_mfr(rows["train"], lang) if isinstance(rows, dict) else build_mfr(rows, lang)

    eval_rows = rows["valid"] if isinstance(rows, dict) else rows
    train_rows = rows["train"] if isinstance(rows, dict) else rows
    mfr, mfr_conf = build_mfr(train_rows, lang)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    raw_sents, gold_sents, pred_sents = [], [], []
    counts = Counter()

    iterator = tqdm(eval_rows, desc=f"hybrid {lang}") if show_progress else eval_rows
    for row in iterator:
        if row["lang"] != lang:
            continue
        raw_words = row["raw"]
        gold_words = [target_of(r, n) for r, n in zip(row["raw"], row["norm"])]
        pred_words = []
        for i, raw in enumerate(raw_words):
            if is_protected_token(raw):
                pred_words.append(raw)
                counts["protected_copy"] += 1
                continue

            mfr_pred = mfr.get(raw, raw)
            if mfr_pred != raw and mfr_conf.get(raw, 0.0) >= mfr_min_conf:
                pred_words.append(mfr_pred)
                counts["mfr_change"] += 1
                continue

            p_change = detector_prob(detector, raw_words, i)
            if p_change >= threshold:
                pred = predict_byt5_word(tokenizer, model, device, lang, raw, raw_words)
                if pred:
                    pred_words.append(pred)
                    counts["byt5_requested"] += 1
                    if pred != raw:
                        counts["byt5_change"] += 1
                    else:
                        counts["byt5_copy"] += 1
                else:
                    pred_words.append(raw)
                    counts["byt5_empty_copy"] += 1
            else:
                pred_words.append(raw)
                counts["detector_copy"] += 1

        raw_sents.append(raw_words)
        gold_sents.append(gold_words)
        pred_sents.append(pred_words)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return raw_sents, gold_sents, pred_sents, counts


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def cmd_train_final(args):
    set_seed(args.seed)
    rows = load_train_rows(args.train_file, args.dataset_name)
    os.makedirs(args.final_model_dir, exist_ok=True)
    os.makedirs(args.detector_dir, exist_ok=True)

    if args.train_western:
        for lang in args.western_langs.split(","):
            lang = lang.strip()
            if not lang:
                continue
            fmt = WESTERN_STRATEGY.get(lang, "natural")
            base_model = f"ufal/byt5-small-multilexnorm2021-{lang}"
            print(f"\n[{lang.upper()}] western ByT5 training, format={fmt}, base={base_model}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                _ = AutoModelForSeq2SeqLM.from_pretrained(base_model)
                del _
            except Exception as e:
                print(f"skip {lang}: cannot load {base_model}: {e}")
                continue
            ds = WesternByT5Dataset(rows, lang, tokenizer, max_length=args.max_length)
            epochs = args.western_natural_epochs if fmt == "natural" else args.western_sentinel_epochs
            train_seq2seq_model(
                ds,
                base_model,
                os.path.join(args.final_model_dir, f"{lang}_model"),
                epochs=epochs,
                lr=args.western_lr,
                batch_size=args.batch_size,
                bf16=args.bf16,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                seed=args.seed,
            )

    if args.train_es_it:
        for lang in args.es_it_langs.split(","):
            lang = lang.strip()
            if not lang:
                continue
            base_model = args.es_it_base_model
            print(f"\n[{lang.upper()}] ES/IT ByT5 normalizer training, base={base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            ds = EsItByT5Dataset(
                rows,
                lang,
                tokenizer,
                max_length=args.max_length,
                unchanged_keep_prob=args.unchanged_keep_prob,
                seed=args.seed,
            )
            train_seq2seq_model(
                ds,
                base_model,
                os.path.join(args.final_model_dir, f"{lang}_model"),
                epochs=args.es_it_epochs,
                lr=args.es_it_lr,
                batch_size=args.batch_size,
                bf16=args.bf16,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                seed=args.seed,
            )

            detector_output = os.path.join(args.detector_dir, f"{lang}_change_detector_rf.joblib")
            train_change_detector(rows, lang, detector_output, args)


def split_lang_rows(rows, lang: str, test_size: float, seed: int):
    lang_rows = rows.filter(lambda x: x["lang"] == lang)
    split = lang_rows.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def cmd_eval_es_it(args):
    set_seed(args.seed)
    rows = load_train_rows(args.train_file, args.dataset_name)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, "detectors"), exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, "reports"), exist_ok=True)

    for lang in args.es_it_langs.split(","):
        lang = lang.strip()
        train_rows, valid_rows = split_lang_rows(rows, lang, args.test_size, args.seed)
        print(f"\n[{lang.upper()} holdout] train rows={len(train_rows)} valid rows={len(valid_rows)}")

        model_dir = os.path.join(args.eval_dir, "models", f"{lang}_model")
        detector_path = os.path.join(args.eval_dir, "detectors", f"{lang}_change_detector_rf.joblib")

        tokenizer = AutoTokenizer.from_pretrained(args.es_it_base_model)
        ds = EsItByT5Dataset(
            train_rows,
            lang,
            tokenizer,
            max_length=args.max_length,
            unchanged_keep_prob=args.unchanged_keep_prob,
            seed=args.seed,
        )
        train_seq2seq_model(
            ds,
            args.es_it_base_model,
            model_dir,
            epochs=args.es_it_epochs,
            lr=args.es_it_lr,
            batch_size=args.batch_size,
            bf16=args.bf16,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            seed=args.seed,
        )

        train_change_detector(train_rows, lang, detector_path, args)

        mfr, _ = build_mfr(train_rows, lang)
        raw_sents, gold_sents, mfr_preds = [], [], []
        for row in valid_rows:
            raw_words = row["raw"]
            gold_words = [target_of(r, n) for r, n in zip(row["raw"], row["norm"])]
            raw_sents.append(raw_words)
            gold_sents.append(gold_words)
            mfr_preds.append([mfr.get(w, w) for w in raw_words])
        print_metrics(f"{lang.upper()} MFR baseline", evaluate_predictions(raw_sents, gold_sents, mfr_preds))

        threshold = args.es_threshold if lang == "es" else args.it_threshold
        mfr_min_conf = args.mfr_min_conf
        raw_h, gold_h, pred_h, counts = predict_hybrid(
            {"train": train_rows, "valid": valid_rows},
            lang,
            detector_path,
            model_dir,
            threshold,
            mfr_min_conf,
        )
        print("\nHybrid decision counts:")
        for k, v in counts.most_common():
            print(f"  {k:24s} {v}")
        metrics = evaluate_predictions(raw_h, gold_h, pred_h)
        print_metrics(f"{lang.upper()} RF + MFR + ByT5 hybrid", metrics, show_errors=args.verbose)


def add_common_train_args(p):
    p.add_argument("--train_file", type=str, default="./data/train-00000-of-00001.parquet")
    p.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", dest="bf16", action="store_false")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    p.add_argument("--es_it_base_model", type=str, default="google/byt5-small")
    p.add_argument("--es_it_epochs", type=float, default=5)
    p.add_argument("--es_it_lr", type=float, default=1e-5)
    p.add_argument("--unchanged_keep_prob", type=float, default=0.8)
    p.add_argument("--es_it_langs", type=str, default="es,it")

    p.add_argument("--detector_n_estimators", type=int, default=DEFAULT_DETECTOR_PARAMS["n_estimators"])
    p.add_argument("--detector_max_depth", type=int, default=DEFAULT_DETECTOR_PARAMS["max_depth"])
    p.add_argument("--detector_min_samples_leaf", type=int, default=DEFAULT_DETECTOR_PARAMS["min_samples_leaf"])
    p.add_argument("--detector_min_samples_split", type=int, default=DEFAULT_DETECTOR_PARAMS["min_samples_split"])
    p.add_argument("--detector_max_features", type=str, default=DEFAULT_DETECTOR_PARAMS["max_features"])
    p.add_argument("--detector_class_weight", choices=["none", "custom", "balanced", "balanced_subsample"], default=DEFAULT_DETECTOR_PARAMS["class_weight"])
    p.add_argument("--detector_change_weight", type=float, default=DEFAULT_DETECTOR_PARAMS["change_weight"])
    p.add_argument("--n_jobs", type=int, default=-1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train_final")
    add_common_train_args(p_train)
    p_train.add_argument("--final_model_dir", type=str, default="./final_model")
    p_train.add_argument("--detector_dir", type=str, default="./detectors")
    p_train.add_argument("--train_western", action="store_true", default=True)
    p_train.add_argument("--no_train_western", dest="train_western", action="store_false")
    p_train.add_argument("--train_es_it", action="store_true", default=True)
    p_train.add_argument("--no_train_es_it", dest="train_es_it", action="store_false")
    p_train.add_argument("--western_langs", type=str, default=",".join(WESTERN_STRATEGY.keys()))
    p_train.add_argument("--western_lr", type=float, default=3e-5)
    p_train.add_argument("--western_natural_epochs", type=float, default=1)
    p_train.add_argument("--western_sentinel_epochs", type=float, default=2)
    p_train.set_defaults(func=cmd_train_final)

    p_eval = sub.add_parser("eval_es_it")
    add_common_train_args(p_eval)
    p_eval.add_argument("--eval_dir", type=str, default="./eval_runs/es_it_hybrid")
    p_eval.add_argument("--test_size", type=float, default=0.2)
    p_eval.add_argument("--es_threshold", type=float, default=DEFAULT_HYBRID["es"]["threshold"])
    p_eval.add_argument("--it_threshold", type=float, default=DEFAULT_HYBRID["it"]["threshold"])
    p_eval.add_argument("--mfr_min_conf", type=float, default=0.65)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_eval_es_it)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
