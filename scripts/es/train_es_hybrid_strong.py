#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/es/train_es_hybrid_strong.py

Train Spanish strong lexical normalizer:
- ES ByT5 normalizer
- RF change detector
- RF candidate ranker
- resources for MFR/key/ngram/candidates
"""

import argparse
import gc
import os
import random
from collections import Counter

import joblib
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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

from es_rules import (
    build_stats,
    candidate_features,
    detector_features,
    generate_candidates,
    is_protected_token,
    target_of,
)

LANG = "es"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(train_file, dataset_name):
    if train_file and os.path.exists(train_file):
        print(f"Loading local train file: {train_file}")
        return load_dataset("parquet", data_files={"train": train_file})["train"]
    print(f"Loading HF dataset: {dataset_name}")
    return load_dataset(dataset_name, split="train")


def split_lang_rows(rows, valid_ratio, seed):
    es_rows = [row for row in rows if row["lang"] == LANG]
    rng = random.Random(seed)
    rng.shuffle(es_rows)
    if valid_ratio <= 0:
        return es_rows, []
    n_valid = max(1, int(len(es_rows) * valid_ratio))
    return es_rows[n_valid:], es_rows[:n_valid]


class EsByT5Dataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128, unchanged_keep_prob=0.8, changed_repeat=3, seed=5, prompt_format="sentinel"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        rng = random.Random(seed)
        changed = 0
        unchanged = 0

        for row in rows:
            if row["lang"] != LANG:
                continue
            raw_words = list(row["raw"])
            norm_words = list(row["norm"])
            context = " ".join(raw_words)

            for i, raw in enumerate(raw_words):
                target = target_of(raw, norm_words[i])
                is_changed = raw != target
                if not is_changed and rng.random() > unchanged_keep_prob:
                    continue

                if prompt_format == "natural":
                    input_text = f"lang: es word: {raw} context: {context}"
                elif prompt_format == "marked_natural":
                    words_copy = list(raw_words)
                    words_copy[i] = f"<extra_id_0> {raw} <extra_id_1>"
                    marked_context = " ".join(words_copy)
                    input_text = f"normalize lang: es target: {raw} context: {marked_context}"
                else:
                    words_copy = list(raw_words)
                    words_copy[i] = f"<extra_id_0> {raw} <extra_id_1>"
                    input_text = " ".join(words_copy)

                repeat = changed_repeat if is_changed else 1
                for _ in range(repeat):
                    self.samples.append({"input_text": input_text, "target_text": target})
                if is_changed:
                    changed += repeat
                else:
                    unchanged += 1

        print(f"[ES ByT5 dataset] samples={len(self.samples)} changed_repeated={changed} unchanged={unchanged}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        model_inputs = self.tokenizer(s["input_text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(s["target_text"], max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


def choose_base_model(preferred):
    candidates = [preferred, "ufal/byt5-small-multilexnorm2021-es", "google/byt5-small"]
    for name in candidates:
        try:
            AutoTokenizer.from_pretrained(name)
            AutoModelForSeq2SeqLM.from_pretrained(name)
            print(f"Using base model: {name}")
            return name
        except Exception as e:
            print(f"Failed base model: {name} ({e})")
    raise RuntimeError("No usable ES base model.")


def train_byt5(rows, args):
    if args.skip_byt5:
        print("Skip ByT5 training.")
        return

    base_model_name = choose_base_model(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dataset = EsByT5Dataset(
        rows=rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        unchanged_keep_prob=args.unchanged_keep_prob,
        changed_repeat=args.changed_repeat,
        seed=args.seed,
        prompt_format=args.prompt_format,
    )
    if len(dataset) == 0:
        raise RuntimeError("Empty ES ByT5 dataset.")

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
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)
    print(f"Saved ES ByT5 model: {args.output_model_dir}")

    del model, trainer, tokenizer, merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def build_detector_examples(rows, resources):
    X, y = [], []
    for row in rows:
        if row["lang"] != LANG:
            continue
        raw_words = list(row["raw"])
        norm_words = list(row["norm"])
        for i, raw in enumerate(raw_words):
            gold = target_of(raw, norm_words[i])
            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
            X.append(detector_features(raw, left, right, resources))
            y.append(int(raw != gold))
    return X, np.array(y, dtype=np.int64)


def train_detector(rows, resources, args):
    X, y = build_detector_examples(rows, resources)
    print("\n[ES RF change detector]")
    print("examples:", len(y))
    print("COPY:", int((y == 0).sum()))
    print("CHANGE:", int((y == 1).sum()))
    print("CHANGE rate:", float(y.mean()) if len(y) else 0.0)

    if args.detector_class_weight == "none":
        class_weight = None
    elif args.detector_class_weight == "custom":
        class_weight = {0: 1.0, 1: args.detector_change_weight}
    else:
        class_weight = args.detector_class_weight

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
    pred = clf.predict(X)
    print("\n[Train detector report]")
    print(classification_report(y, pred, target_names=["COPY", "CHANGE"], digits=4))
    print(confusion_matrix(y, pred))

    artifact = {"model": clf, "threshold": args.detector_threshold, "lang": LANG, "params": vars(args)}
    os.makedirs(os.path.dirname(args.output_detector), exist_ok=True)
    joblib.dump(artifact, args.output_detector)
    print(f"Saved detector: {args.output_detector}")


def build_ranker_examples(rows, resources, include_gold=True):
    X, y = [], []
    upper_total = upper_hit = changed_total = changed_hit = 0
    source_hits = Counter()
    mfr = resources["mfr"]
    mfr_conf = resources["mfr_conf"]
    key_map = resources["key_map"]

    for row in tqdm(rows, desc="Build ranker examples"):
        if row["lang"] != LANG:
            continue
        raw_words = list(row["raw"])
        norm_words = list(row["norm"])
        for i, raw in enumerate(raw_words):
            gold = target_of(raw, norm_words[i])
            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
            cands = generate_candidates(raw, mfr=mfr, mfr_conf=mfr_conf, key_map=key_map)
            cand_values = [c for _s, c in cands]
            upper_total += 1
            if raw != gold:
                changed_total += 1
            if gold in cand_values:
                upper_hit += 1
                if raw != gold:
                    changed_hit += 1
                for source, cand in cands:
                    if cand == gold:
                        source_hits[source] += 1
                        break
            elif include_gold:
                cands.append(("gold_forcing", gold))
            for source, cand in cands:
                X.append(candidate_features(raw, cand, source, left, right, resources))
                y.append(int(cand == gold))

    print("\n[Candidate upperbound before gold-forcing]")
    print(f"candidate upperbound: {upper_hit / upper_total * 100 if upper_total else 0:.2f}%")
    print(f"changed candidate upperbound: {changed_hit / changed_total * 100 if changed_total else 0:.2f}%")
    print("source gold hits:")
    for k, v in source_hits.most_common():
        print(f"  {k:20s} {v}")
    return X, np.array(y, dtype=np.int64)


def train_ranker(rows, resources, args):
    X, y = build_ranker_examples(rows, resources, include_gold=True)
    print("\n[ES RF candidate ranker]")
    print("examples:", len(y))
    print("positive:", int((y == 1).sum()))
    print("negative:", int((y == 0).sum()))

    if args.ranker_class_weight == "none":
        class_weight = None
    elif args.ranker_class_weight == "custom":
        class_weight = {0: 1.0, 1: args.ranker_positive_weight}
    else:
        class_weight = args.ranker_class_weight

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=args.ranker_n_estimators,
            max_depth=args.ranker_max_depth if args.ranker_max_depth > 0 else None,
            min_samples_leaf=args.ranker_min_samples_leaf,
            min_samples_split=args.ranker_min_samples_split,
            max_features=args.ranker_max_features,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )),
    ])
    clf.fit(X, y)
    pred = clf.predict(X)
    print("\n[Train ranker report]")
    print(classification_report(y, pred, target_names=["WRONG", "GOLD"], digits=4))
    print(confusion_matrix(y, pred))

    artifact = {"model": clf, "ranker_threshold": args.ranker_threshold, "lang": LANG, "params": vars(args)}
    os.makedirs(os.path.dirname(args.output_ranker), exist_ok=True)
    joblib.dump(artifact, args.output_ranker)
    print(f"Saved ranker: {args.output_ranker}")


def evaluate(raw_sents, gold_sents, pred_sents):
    total = raw_correct = pred_correct = 0
    changed_total = changed_correct = 0
    unchanged_total = unchanged_preserve = 0
    errors = []
    for raw_words, gold_words, pred_words in zip(raw_sents, gold_sents, pred_sents):
        for raw, gold, pred in zip(raw_words, gold_words, pred_words):
            total += 1
            raw_correct += int(raw == gold)
            pred_correct += int(pred == gold)
            if raw != gold:
                changed_total += 1
                changed_correct += int(pred == gold)
            else:
                unchanged_total += 1
                unchanged_preserve += int(pred == raw)
            if pred != gold and len(errors) < 80:
                errors.append((raw, gold, pred))
    baseline = raw_correct / total * 100 if total else 0.0
    acc = pred_correct / total * 100 if total else 0.0
    err = ((acc - baseline) / (100 - baseline) * 100) if baseline < 100 else 0.0
    changed_acc = changed_correct / changed_total * 100 if changed_total else 0.0
    unchanged_pres = unchanged_preserve / unchanged_total * 100 if unchanged_total else 0.0
    print(f"Total tokens:              {total}")
    print(f"Changed tokens:            {changed_total}")
    print(f"Changed rate:              {changed_total / total * 100 if total else 0:.2f}%")
    print(f"Baseline acc.(LAI):        {baseline:.2f}")
    print(f"Accuracy:                  {acc:.2f}")
    print(f"ERR:                       {err:.2f}")
    print(f"Changed token accuracy:    {changed_acc:.2f}")
    print(f"Unchanged preservation:    {unchanged_pres:.2f}")
    print(f"Over-change rate:          {100 - unchanged_pres:.2f}")
    print("\nError examples: raw -> gold / pred")
    for r, g, p in errors:
        print(f"{r} -> {g} / {p}")


def eval_holdout(valid_rows, resources, args):
    print("\n[ES holdout eval: detector + ranker]")
    detector = joblib.load(args.output_detector)["model"]
    ranker = joblib.load(args.output_ranker)["model"]
    raw_sents, gold_sents, pred_sents = [], [], []
    counts = Counter()

    for row in tqdm(valid_rows, desc="Eval ES"):
        raw_words = list(row["raw"])
        gold_words = [target_of(r, n) for r, n in zip(row["raw"], row["norm"])]
        pred_words = list(raw_words)
        for i, raw in enumerate(raw_words):
            if is_protected_token(raw):
                counts["protected_copy"] += 1
                continue
            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
            dprob = detector.predict_proba([detector_features(raw, left, right, resources)])[0][1]
            if dprob < args.detector_threshold:
                counts["detector_copy"] += 1
                continue
            cands = generate_candidates(raw, mfr=resources["mfr"], mfr_conf=resources["mfr_conf"], key_map=resources["key_map"])
            feats = [candidate_features(raw, cand, source, left, right, resources) for source, cand in cands]
            probs = ranker.predict_proba(feats)[:, 1]
            best_i = int(np.argmax(probs))
            best_source, best_cand = cands[best_i]
            best_score = probs[best_i]
            if best_cand != raw and best_score >= args.ranker_threshold:
                pred_words[i] = best_cand
                counts["ranker_change"] += 1
                counts[f"ranker_source_{best_source}"] += 1
            else:
                counts["ranker_copy"] += 1
        raw_sents.append(raw_words)
        gold_sents.append(gold_words)
        pred_sents.append(pred_words)
    print("Decision counts:")
    for k, v in counts.most_common():
        print(f"  {k:28s} {v}")
    evaluate(raw_sents, gold_sents, pred_sents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./data/train-00000-of-00001.parquet")
    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--output_model_dir", type=str, default="./final_model/es_model")
    parser.add_argument("--output_detector", type=str, default="./detectors/es_change_detector_rf.joblib")
    parser.add_argument("--output_ranker", type=str, default="./detectors/es_candidate_ranker_rf.joblib")
    parser.add_argument("--output_resources", type=str, default="./detectors/es_resources.joblib")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--base_model", type=str, default="ufal/byt5-small-multilexnorm2021-es")
    parser.add_argument("--prompt_format", choices=["sentinel", "natural", "marked_natural"], default="sentinel")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--unchanged_keep_prob", type=float, default=0.8)
    parser.add_argument("--changed_repeat", type=int, default=3)
    parser.add_argument("--skip_byt5", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--detector_n_estimators", type=int, default=600)
    parser.add_argument("--detector_max_depth", type=int, default=12)
    parser.add_argument("--detector_min_samples_leaf", type=int, default=1)
    parser.add_argument("--detector_min_samples_split", type=int, default=4)
    parser.add_argument("--detector_max_features", type=str, default="sqrt")
    parser.add_argument("--detector_class_weight", choices=["none", "custom", "balanced", "balanced_subsample"], default="custom")
    parser.add_argument("--detector_change_weight", type=float, default=3.5)
    parser.add_argument("--detector_threshold", type=float, default=0.48)
    parser.add_argument("--ranker_n_estimators", type=int, default=700)
    parser.add_argument("--ranker_max_depth", type=int, default=14)
    parser.add_argument("--ranker_min_samples_leaf", type=int, default=1)
    parser.add_argument("--ranker_min_samples_split", type=int, default=4)
    parser.add_argument("--ranker_max_features", type=str, default="sqrt")
    parser.add_argument("--ranker_class_weight", choices=["none", "custom", "balanced", "balanced_subsample"], default="custom")
    parser.add_argument("--ranker_positive_weight", type=float, default=2.0)
    parser.add_argument("--ranker_threshold", type=float, default=0.45)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_rows(args.train_file, args.dataset_name)
    train_rows, valid_rows = split_lang_rows(rows, args.valid_ratio, args.seed)
    print(f"[ES] train rows: {len(train_rows)}")
    print(f"[ES] valid rows: {len(valid_rows)}")

    resources = build_stats(train_rows, LANG)
    os.makedirs(os.path.dirname(args.output_resources), exist_ok=True)
    joblib.dump(resources, args.output_resources)
    print(f"Saved resources: {args.output_resources}")

    train_byt5(train_rows, args)
    train_detector(train_rows, resources, args)
    train_ranker(train_rows, resources, args)

    if args.valid_ratio > 0 and len(valid_rows) > 0:
        eval_holdout(valid_rows, resources, args)


if __name__ == "__main__":
    main()
