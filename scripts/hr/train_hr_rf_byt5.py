#!/usr/bin/env python3
"""
Train Croatian(HR) two-stage normalizer:
1) RandomForest token-change classifier: raw token -> should this token change?
2) ByT5 seq2seq normalizer: applied only to tokens predicted as changed at submission time.

Outputs by default:
- final_model/hr_change_rf.joblib
- final_model/hr_model/
"""

import argparse
import gc
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "peft is required for LoRA training. Install it with: pip install peft"
    ) from exc

LANG = "hr"
DEFAULT_BASE_MODEL = "ufal/byt5-small-multilexnorm2021-hr"


def normalize_gold(raw: str, norm: Optional[str]) -> str:
    return raw if norm is None else norm


def build_mfr_dictionary(rows: Sequence[dict]) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = normalize_gold(raw, norm)
            counts[raw][target] += 1
    # tie-breaker: prefer keeping the original token if frequency is tied
    return {
        raw: max(targets.items(), key=lambda item: (item[1], item[0] == raw))[0]
        for raw, targets in counts.items()
    }


def _safe_token(tokens: Sequence[str], idx: int) -> str:
    if 0 <= idx < len(tokens):
        tok = tokens[idx]
        return "" if tok is None else str(tok)
    return "<BOS>" if idx < 0 else "<EOS>"


def token_features(raw_words: Sequence[str], i: int, mfr_dict: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Feature set shared by training and submission.

    It intentionally uses only information available at test time: current token,
    local context, shape features, and MFR statistics derived from the training split.
    """
    word = _safe_token(raw_words, i)
    lower = word.lower()
    prev1 = _safe_token(raw_words, i - 1).lower()
    prev2 = _safe_token(raw_words, i - 2).lower()
    next1 = _safe_token(raw_words, i + 1).lower()
    next2 = _safe_token(raw_words, i + 2).lower()

    n_chars = max(len(word), 1)
    n_alpha = sum(ch.isalpha() for ch in word)
    n_digit = sum(ch.isdigit() for ch in word)
    n_upper = sum(ch.isupper() for ch in word)
    n_punct = sum((not ch.isalnum()) for ch in word)
    repeated_char = any(word[j] == word[j - 1] for j in range(1, len(word)))

    mfr_pred = mfr_dict.get(word, word) if mfr_dict is not None else word

    feats: Dict[str, object] = {
        "bias": 1,
        "tok=" + lower: 1,
        "prev1=" + prev1: 1,
        "prev2=" + prev2: 1,
        "next1=" + next1: 1,
        "next2=" + next2: 1,
        "prev_bigram=" + prev1 + "|" + lower: 1,
        "next_bigram=" + lower + "|" + next1: 1,
        "len": len(word),
        "sent_len": len(raw_words),
        "pos_abs": i,
        "pos_rel": i / max(len(raw_words) - 1, 1),
        "is_lower": int(word.islower()),
        "is_upper": int(word.isupper()),
        "is_title": int(word.istitle()),
        "has_alpha": int(n_alpha > 0),
        "has_digit": int(n_digit > 0),
        "has_punct": int(n_punct > 0),
        "has_at": int("@" in word),
        "has_hash": int("#" in word),
        "has_url_piece": int("http" in lower or "www" in lower or ".com" in lower),
        "repeated_char": int(repeated_char),
        "digit_ratio": n_digit / n_chars,
        "upper_ratio": n_upper / n_chars,
        "punct_ratio": n_punct / n_chars,
        "alpha_ratio": n_alpha / n_chars,
        "mfr_seen": int(mfr_dict is not None and word in mfr_dict),
        "mfr_would_change": int(mfr_pred != word),
        "mfr_pred=" + mfr_pred.lower(): 1,
    }

    for k in range(1, 5):
        if len(lower) >= k:
            feats[f"pref{k}=" + lower[:k]] = 1
            feats[f"suf{k}=" + lower[-k:]] = 1

    return feats


def build_rf_examples(rows: Sequence[dict], mfr_dict: Dict[str, str]) -> Tuple[List[Dict[str, object]], List[int]]:
    x: List[Dict[str, object]] = []
    y: List[int] = []
    for row in rows:
        raw_words = row["raw"]
        norm_words = row["norm"]
        for i, (raw, norm) in enumerate(zip(raw_words, norm_words)):
            target = normalize_gold(raw, norm)
            x.append(token_features(raw_words, i, mfr_dict))
            y.append(int(target != raw))
    return x, y


def split_rows(rows: Sequence[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if val_ratio <= 0 or len(rows) < 10:
        return rows, []
    n_val = max(1, int(round(len(rows) * val_ratio)))
    return rows[n_val:], rows[:n_val]


def tune_threshold(y_true: Sequence[int], proba: Sequence[float], metric: str = "f1") -> Tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    # Conservative range: very low thresholds over-normalize too aggressively.
    grid = [round(x / 100, 2) for x in range(20, 91, 2)]
    for threshold in grid:
        pred = [int(p >= threshold) for p in proba]
        if metric == "precision":
            precision, _, _, _ = precision_recall_fscore_support(
                y_true, pred, average="binary", zero_division=0
            )
            score = float(precision)
        else:
            score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


class HrByT5Dataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        tokenizer_name: str,
        max_length: int = 128,
        target_max_length: int = 64,
        changed_only: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.samples: List[Dict[str, str]] = []

        for row in rows:
            raw_words = row["raw"]
            norm_words = row["norm"]
            context = " ".join(raw_words)
            for raw, norm in zip(raw_words, norm_words):
                target = normalize_gold(raw, norm)
                if changed_only and target == raw:
                    continue
                input_text = f"lang: {LANG} word: {raw} context: {context}"
                self.samples.append({"input_text": input_text, "target_text": target})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
            max_length=self.target_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": label_ids,
        }



def parse_optional_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "-1"}:
        return None
    return int(text)


def parse_optional_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "-1"}:
        return None
    return float(text)


def parse_rf_max_features(value: Optional[object]) -> Optional[object]:
    """Parse sklearn RandomForest max_features from CLI/bash-friendly text.

    Supported examples: none, sqrt, log2, 0.5, 128.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "-1"}:
        return None
    if text in {"sqrt", "log2"}:
        return text
    if "." in text:
        return float(text)
    return int(text)


def parse_rf_class_weight(value: Optional[object]) -> Optional[str]:
    if value is None:
        return "balanced_subsample"
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text not in {"balanced", "balanced_subsample"}:
        raise ValueError("--rf_class_weight must be one of: none, balanced, balanced_subsample")
    return text


def train_rf(args: argparse.Namespace, rows: Sequence[dict], model_dir: str) -> None:
    print("\n==============================")
    print("1. Train HR RandomForest change classifier")
    print("==============================")

    os.makedirs(model_dir, exist_ok=True)
    mfr_dict = build_mfr_dictionary(rows)
    train_rows, val_rows = split_rows(rows, args.rf_val_ratio, args.seed)

    x_train, y_train = build_rf_examples(train_rows, mfr_dict)
    print(f"RF train tokens: {len(y_train):,} | changed={sum(y_train):,} | unchanged={len(y_train)-sum(y_train):,}")

    rf_params = {
        "n_estimators": args.rf_n_estimators,
        "criterion": args.rf_criterion,
        "max_depth": parse_optional_int(args.rf_max_depth),
        "min_samples_split": args.rf_min_samples_split,
        "min_samples_leaf": args.rf_min_samples_leaf,
        "max_features": parse_rf_max_features(args.rf_max_features),
        "max_leaf_nodes": parse_optional_int(args.rf_max_leaf_nodes),
        "bootstrap": bool(args.rf_bootstrap),
        "max_samples": parse_optional_float(args.rf_max_samples),
        "class_weight": parse_rf_class_weight(args.rf_class_weight),
        "n_jobs": args.rf_n_jobs,
        "random_state": args.seed,
        "verbose": args.rf_verbose,
    }
    if not rf_params["bootstrap"]:
        # sklearn requires max_samples=None when bootstrap=False.
        rf_params["max_samples"] = None

    print("RF params:")
    for key in sorted(rf_params):
        print(f"  {key}: {rf_params[key]}")

    rf = RandomForestClassifier(**rf_params)
    pipeline = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("rf", rf),
    ])
    pipeline.fit(x_train, y_train)

    threshold = args.rf_threshold
    val_report = None
    if val_rows:
        x_val, y_val = build_rf_examples(val_rows, mfr_dict)
        proba = pipeline.predict_proba(x_val)[:, 1]
        threshold, best_score = tune_threshold(y_val, proba, args.rf_threshold_metric)
        val_pred = [int(p >= threshold) for p in proba]
        val_report = classification_report(y_val, val_pred, digits=4, zero_division=0)
        print(f"RF tuned threshold: {threshold:.2f} | val {args.rf_threshold_metric}: {best_score:.4f}")
        print(val_report)

    if args.refit_rf_full and val_rows:
        print("Refit RF on all HR train rows with the tuned threshold.")
        x_all, y_all = build_rf_examples(rows, mfr_dict)
        pipeline.fit(x_all, y_all)

    out_path = os.path.join(model_dir, "hr_change_rf.joblib")
    dump(
        {
            "pipeline": pipeline,
            "threshold": float(threshold),
            "lang": LANG,
            "feature_version": "hr_rf_byt5_v1",
            "mfr_dict": mfr_dict,
            "args": vars(args),
            "val_report": val_report,
            "label_counter": dict(Counter(y_train)),
        },
        out_path,
    )
    print(f"Saved RF classifier: {out_path}")


def train_byt5(args: argparse.Namespace, rows: Sequence[dict], model_dir: str) -> None:
    print("\n==============================")
    print("2. Train HR ByT5 normalizer")
    print("==============================")

    final_path = os.path.join(model_dir, "hr_model")
    os.makedirs(model_dir, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load base model {args.base_model}. "
            "Check internet/cache or pass --base_model google/byt5-small."
        ) from exc

    dataset = HrByT5Dataset(
        rows=rows,
        tokenizer_name=args.base_model,
        max_length=args.max_length,
        target_max_length=args.target_max_length,
        changed_only=args.byt5_changed_only,
    )
    if len(dataset) == 0:
        raise RuntimeError("No HR training samples were created for ByT5.")
    print(f"ByT5 train samples: {len(dataset):,} | changed_only={args.byt5_changed_only}")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)

    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.work_dir, "models", "byt5-hr-rf-gated"),
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        bf16=bool(args.bf16 and use_cuda),
        fp16=bool(args.fp16 and use_cuda),
        report_to="none",
        logging_steps=args.logging_steps,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved ByT5 model: {final_path}")

    del model, trainer, tokenizer, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--split", default="train")
    parser.add_argument("--work_dir", default=".")
    parser.add_argument("--model_dir", default="final_model")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip_rf", action="store_true")
    parser.add_argument("--rf_n_estimators", type=int, default=500)
    parser.add_argument("--rf_criterion", choices=["gini", "entropy", "log_loss"], default="gini")
    parser.add_argument("--rf_max_depth", default="none", help="int or none")
    parser.add_argument("--rf_min_samples_split", type=int, default=2)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=2)
    parser.add_argument("--rf_max_features", default="none", help="none, sqrt, log2, float ratio, or int")
    parser.add_argument("--rf_max_leaf_nodes", default="none", help="int or none")
    parser.add_argument("--rf_bootstrap", type=int, default=1)
    parser.add_argument("--rf_max_samples", default="none", help="none, float ratio, or int; valid only when bootstrap=1")
    parser.add_argument("--rf_class_weight", default="balanced_subsample", help="none, balanced, or balanced_subsample")
    parser.add_argument("--rf_n_jobs", type=int, default=-1)
    parser.add_argument("--rf_verbose", type=int, default=0)
    parser.add_argument("--rf_val_ratio", type=float, default=0.10)
    parser.add_argument("--rf_threshold", type=float, default=0.50)
    parser.add_argument("--rf_threshold_metric", choices=["f1", "precision"], default="f1")
    parser.add_argument("--refit_rf_full", type=int, default=1)

    parser.add_argument("--skip_byt5", action="store_true")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--target_max_length", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.10)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--fp16", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--byt5_changed_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    model_dir = os.path.join(args.work_dir, args.model_dir)
    full_dataset = load_dataset(args.dataset)
    rows = [row for row in full_dataset[args.split] if row["lang"] == LANG]
    if not rows:
        raise RuntimeError(f"No rows found for lang={LANG} in split={args.split}")

    print(f"Loaded HR rows: {len(rows):,} from {args.dataset}/{args.split}")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "hr_train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    if not args.skip_rf:
        train_rf(args, rows, model_dir)
    if not args.skip_byt5:
        train_byt5(args, rows, model_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
