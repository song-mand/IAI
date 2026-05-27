# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from th_utils import (
    HANGING_VOWELS,
    LEADING_VOWELS,
    TONE_MARKS,
    env_bool,
    seed_everything,
    target_or_raw,
    thai_normalize_token,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.environ.get("DATASET", "weerayut/multilexnorm2026-dev-pub"))
    p.add_argument("--split", default=os.environ.get("TRAIN_SPLIT", "train"))
    p.add_argument("--lang", default=os.environ.get("LANG", "th"))
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "google/byt5-small"))
    p.add_argument("--output-dir", default=os.environ.get("TH_BYT5_OUTPUT", "final_model/th_byt5_candidate"))
    p.add_argument("--work-dir", default=os.environ.get("TH_BYT5_WORK", "models/th/byt5_work"))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))

    # Data construction
    p.add_argument("--keep-unchanged-prob", type=float, default=float(os.environ.get("KEEP_UNCHANGED_PROB", "0.08")))
    p.add_argument("--aug-per-token", type=int, default=int(os.environ.get("AUG_PER_TOKEN", "2")))
    p.add_argument("--max-real-samples", type=int, default=int(os.environ.get("MAX_REAL_SAMPLES", "0")))
    p.add_argument("--max-aug-samples", type=int, default=int(os.environ.get("MAX_AUG_SAMPLES", "0")))
    p.add_argument("--input-format", default=os.environ.get("INPUT_FORMAT", "natural"), choices=["natural", "sentinel"])

    # Noise probabilities
    p.add_argument("--p-delete", type=float, default=float(os.environ.get("P_DELETE", "0.03")))
    p.add_argument("--p-insert", type=float, default=float(os.environ.get("P_INSERT", "0.03")))
    p.add_argument("--p-substitute", type=float, default=float(os.environ.get("P_SUBSTITUTE", "0.03")))
    p.add_argument("--p-repeat-mark", type=float, default=float(os.environ.get("P_REPEAT_MARK", "0.06")))
    p.add_argument("--p-repeat-char", type=float, default=float(os.environ.get("P_REPEAT_CHAR", "0.02")))
    p.add_argument("--p-swap-mark", type=float, default=float(os.environ.get("P_SWAP_MARK", "0.04")))

    # Training hyperparameters
    p.add_argument("--pretrain-aug", type=int, default=int(os.environ.get("PRETRAIN_AUG", "1")))
    p.add_argument("--finetune-real", type=int, default=int(os.environ.get("FINETUNE_REAL", "1")))
    p.add_argument("--aug-epochs", type=float, default=float(os.environ.get("AUG_EPOCHS", "1")))
    p.add_argument("--real-epochs", type=float, default=float(os.environ.get("REAL_EPOCHS", "2")))
    p.add_argument("--aug-lr", type=float, default=float(os.environ.get("AUG_LR", "5e-5")))
    p.add_argument("--real-lr", type=float, default=float(os.environ.get("REAL_LR", "3e-5")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "1")))
    p.add_argument("--max-input-length", type=int, default=int(os.environ.get("MAX_INPUT_LENGTH", "160")))
    p.add_argument("--max-target-length", type=int, default=int(os.environ.get("MAX_TARGET_LENGTH", "64")))
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    p.add_argument("--fp16", type=int, default=int(os.environ.get("FP16", "0")))
    p.add_argument("--bf16", type=int, default=int(os.environ.get("BF16", "1")))

    # LoRA
    p.add_argument("--use-lora", type=int, default=int(os.environ.get("USE_LORA", "1")))
    p.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "16")))
    p.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    p.add_argument("--lora-dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.05")))
    return p.parse_args()


THAI_NOISE_CHARS = list("กขคงจฉชซญดตทนบปผพฟมยรลวสหอเแโใไะาิีึืุูั่้๊๋ๆ")


def corrupt_thai_token(token: str, rng: random.Random, args) -> str:
    """Thai-biased synthetic corruption for ByT5 pretraining."""
    if not token:
        return token
    chars: List[str] = []
    for ch in token:
        # deletion
        if rng.random() < args.p_delete and len(token) > 1:
            continue
        # substitution
        if rng.random() < args.p_substitute:
            chars.append(rng.choice(THAI_NOISE_CHARS))
        else:
            chars.append(ch)
        # insertion
        if rng.random() < args.p_insert:
            chars.append(rng.choice(THAI_NOISE_CHARS))
        # social emphasis: repeated mark or char
        if ch in (TONE_MARKS + HANGING_VOWELS + LEADING_VOWELS) and rng.random() < args.p_repeat_mark:
            chars.append(ch)
        elif rng.random() < args.p_repeat_char:
            chars.append(ch)
    x = "".join(chars)

    # Local mark order corruption: C H T -> C T H, L C T -> L T C.
    if rng.random() < args.p_swap_mark and len(x) >= 3:
        arr = list(x)
        for i in range(len(arr) - 2):
            if arr[i + 1] in HANGING_VOWELS and arr[i + 2] in TONE_MARKS:
                arr[i + 1], arr[i + 2] = arr[i + 2], arr[i + 1]
                break
            if arr[i] in LEADING_VOWELS and arr[i + 2] in TONE_MARKS:
                arr[i + 1], arr[i + 2] = arr[i + 2], arr[i + 1]
                break
        x = "".join(arr)

    # Avoid generating the exact same string too often.
    if x == token and len(token) > 1:
        j = rng.randrange(len(token))
        x = token[:j] + rng.choice(THAI_NOISE_CHARS) + token[j + 1 :]
    return x


def make_input(lang: str, word: str, context: str, index: int, raw_words: List[str], fmt: str) -> str:
    if fmt == "sentinel":
        copied = list(raw_words)
        if 0 <= index < len(copied):
            copied[index] = f"<extra_id_0> {word} <extra_id_1>"
        return " ".join(copied)
    return f"lexnorm lang: {lang} word: {word} context: {context}"


class ThaiByT5Dataset(Dataset):
    def __init__(self, rows, tokenizer, args, mode: str):
        self.tokenizer = tokenizer
        self.args = args
        self.samples: List[Dict[str, str]] = []
        rng = random.Random(args.seed + (17 if mode == "aug" else 0))

        for row in rows:
            if row.get("lang") != args.lang:
                continue
            raw_words = [str(x) for x in row.get("raw", [])]
            norm_words = row.get("norm", raw_words)
            gold_words = [target_or_raw(r, n) for r, n in zip(raw_words, norm_words)]
            context = " ".join(raw_words)

            if mode == "real":
                for i, (raw, gold) in enumerate(zip(raw_words, gold_words)):
                    if raw == gold and rng.random() > args.keep_unchanged_prob:
                        continue
                    inp = make_input(args.lang, raw, context, i, raw_words, args.input_format)
                    self.samples.append({"input_text": inp, "target_text": gold})
            elif mode == "aug":
                for i, gold in enumerate(gold_words):
                    if not gold or len(gold) > 48:
                        continue
                    for _ in range(args.aug_per_token):
                        corrupt = corrupt_thai_token(gold, rng, args)
                        corrupt = thai_normalize_token(corrupt) if rng.random() < 0.15 else corrupt
                        if corrupt == gold:
                            continue
                        aug_words = list(raw_words)
                        if i < len(aug_words):
                            aug_words[i] = corrupt
                        aug_context = " ".join(aug_words)
                        inp = make_input(args.lang, corrupt, aug_context, i, aug_words, args.input_format)
                        self.samples.append({"input_text": inp, "target_text": gold})
            else:
                raise ValueError(mode)

        cap = args.max_real_samples if mode == "real" else args.max_aug_samples
        if cap and len(self.samples) > cap:
            rng.shuffle(self.samples)
            self.samples = self.samples[:cap]
        rng.shuffle(self.samples)
        print(f"[{mode}] samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        model_inputs = self.tokenizer(
            s["input_text"],
            max_length=self.args.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            s["target_text"],
            max_length=self.args.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}


def make_training_args(args, phase: str, lr: float, epochs: float):
    return Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.work_dir, phase),
        overwrite_output_dir=True,
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=epochs,
        dataloader_num_workers=args.num_workers,
        fp16=bool(args.fp16) and torch.cuda.is_available(),
        bf16=bool(args.bf16) and torch.cuda.is_available(),
        report_to="none",
        logging_steps=100,
        predict_with_generate=False,
    )


def make_trainer(model, tokenizer, train_data, training_args):
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    try:
        return Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,
            data_collator=collator,
        )
    except TypeError:
        return Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=tokenizer,
            data_collator=collator,
        )


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    print("=" * 80)
    print("Train Thai ByT5 candidate generator")
    print("=" * 80)
    print(vars(args))

    ds = load_dataset(args.dataset, split=args.split)
    rows = [r for r in ds if r.get("lang") == args.lang]
    print(f"rows={len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules="all-linear",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"[WARN] PEFT/LoRA unavailable, training full model. reason={e}")

    if args.pretrain_aug:
        aug_data = ThaiByT5Dataset(rows, tokenizer, args, mode="aug")
        if len(aug_data) > 0:
            trainer = make_trainer(model, tokenizer, aug_data, make_training_args(args, "aug", args.aug_lr, args.aug_epochs))
            trainer.train()

    if args.finetune_real:
        real_data = ThaiByT5Dataset(rows, tokenizer, args, mode="real")
        if len(real_data) > 0:
            trainer = make_trainer(model, tokenizer, real_data, make_training_args(args, "real", args.real_lr, args.real_epochs))
            trainer.train()

    # Save a normal Transformers model when possible.
    try:
        if hasattr(model, "merge_and_unload"):
            print("Merging LoRA adapter into base model...")
            model = model.merge_and_unload()
    except Exception as e:
        print(f"[WARN] merge_and_unload failed; saving current model. reason={e}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved model: {args.output_dir}")


if __name__ == "__main__":
    main()
