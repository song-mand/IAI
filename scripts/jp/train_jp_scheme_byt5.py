#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Japanese Context-ByT5 for the JP-Copy-MFR-ContextByT5 scheme."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

def repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)

from jp_scheme_common import (
    ArtifactConfig,
    build_artifacts,
    sample_training_examples,
    save_json,
)


class Text2TextListDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_input_length: int, max_target_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        model_inputs = self.tokenizer(
            ex["input_text"],
            max_length=self.max_input_length,
            truncation=True,
        )
        # For ByT5/T5, using the normal tokenizer path for targets is safer
        # across Transformers versions than relying on text_target=.
        labels = self.tokenizer(
            ex["target_text"],
            max_length=self.max_target_length,
            truncation=True,
        )
        label_ids = labels.get("input_ids", [])
        if len(label_ids) == 0:
            eos = self.tokenizer.eos_token_id
            if eos is None:
                raise ValueError("Tokenizer produced an empty target and has no eos_token_id.")
            label_ids = [eos]
        model_inputs["labels"] = label_ids
        return model_inputs


def read_parquet_rows(path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    return df.to_dict("records")


def print_training_sanity(examples: List[Dict[str, str]], tokenizer, max_target_length: int) -> None:
    """Catch the two common causes of loss=0/grad_norm=nan: empty labels and fp16 instability hints."""
    n = min(len(examples), 2000)
    label_lens = []
    empty = 0
    delete_targets = 0
    for ex in examples[:n]:
        ids = tokenizer(ex["target_text"], max_length=max_target_length, truncation=True).get("input_ids", [])
        label_lens.append(len(ids))
        if len(ids) == 0:
            empty += 1
        if ex["target_text"] == "<DELETE>":
            delete_targets += 1
    changed = sum(1 for ex in examples if ex.get("raw") != ex.get("norm"))
    print("[JP][sanity] examples:", len(examples))
    print("[JP][sanity] changed examples:", changed)
    print("[JP][sanity] checked target rows:", n)
    print("[JP][sanity] empty encoded labels:", empty)
    if label_lens:
        print("[JP][sanity] target length min/avg/max:", min(label_lens), round(sum(label_lens) / len(label_lens), 2), max(label_lens))
    print("[JP][sanity] <DELETE> targets in checked rows:", delete_targets)
    print("[JP][sanity] first examples:")
    for ex in examples[:5]:
        print("  RAW=", ex.get("raw"), "NORM=", ex.get("norm"), "TARGET=", ex.get("target_text"), "REASON=", ex.get("reason"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", default=repo_path("train-00000-of-00001.parquet"))
    p.add_argument("--validation_parquet", default=None)
    p.add_argument("--use_validation_for_training", action="store_true")
    p.add_argument("--base_model", default="google/byt5-small")
    p.add_argument("--output_dir", default=repo_path("final_model", "jp_scheme_byt5"))
    p.add_argument("--artifact_path", default=repo_path("final_model", "jp_scheme_artifacts", "jp_scheme_artifacts.json"))
    p.add_argument("--max_input_length", type=int, default=192)
    p.add_argument("--max_target_length", type=int, default=64)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unchanged_sample_rate", type=float, default=0.08)
    p.add_argument("--punct_unchanged_rate", type=float, default=0.50)
    p.add_argument("--min_mfr_count", type=int, default=3)
    p.add_argument("--min_mfr_best_prob", type=float, default=0.75)
    p.add_argument("--min_mfr_change_rate", type=float, default=0.50)
    p.add_argument("--max_mfr_entropy", type=float, default=1.25)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--save_train_examples", action="store_true")
    p.add_argument("--print_train_sanity", action="store_true", help="Print tokenization/label sanity stats before training.")
    p.add_argument("--lang_code", default="ja", help="Dataset language label for Japanese. Uploaded parquet uses ja; set jp only if your data uses jp.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_rows = read_parquet_rows(args.train_parquet)
    artifact_rows = list(train_rows)
    train_for_model = list(train_rows)

    if args.validation_parquet and args.use_validation_for_training:
        valid_rows = read_parquet_rows(args.validation_parquet)
        artifact_rows.extend(valid_rows)
        train_for_model.extend(valid_rows)

    cfg = ArtifactConfig(
        min_mfr_count=args.min_mfr_count,
        min_mfr_best_prob=args.min_mfr_best_prob,
        min_mfr_change_rate=args.min_mfr_change_rate,
        max_mfr_entropy=args.max_mfr_entropy,
    )
    artifacts = build_artifacts(artifact_rows, cfg=cfg, lang=args.lang_code)
    save_json(artifacts, args.artifact_path)
    print(f"[JP] using dataset lang label: {args.lang_code}")
    print("[JP] artifacts saved:", args.artifact_path)
    print("[JP] artifact summary:", artifacts.get("stats_summary"))

    examples = sample_training_examples(
        train_for_model,
        artifacts=artifacts,
        lang=args.lang_code,
        unchanged_sample_rate=args.unchanged_sample_rate,
        punct_unchanged_rate=args.punct_unchanged_rate,
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError(f"No Japanese training examples were created. Check --lang_code={args.lang_code!r} and parquet paths.")
    print(f"[JP] ByT5 training examples: {len(examples):,}")

    if args.save_train_examples:
        import json
        ex_path = os.path.join(os.path.dirname(args.artifact_path), "jp_byt5_train_examples.jsonl")
        os.makedirs(os.path.dirname(ex_path), exist_ok=True)
        with open(ex_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print("[JP] train examples saved:", ex_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print_training_sanity(examples, tokenizer, args.max_target_length)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.config.use_cache = False

    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("--use_lora was set, but peft is not installed. Run: pip install peft")
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    train_dataset = Text2TextListDataset(
        examples,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
    if args.fp16:
        print("[JP][warning] fp16 is enabled. If loss becomes 0 and grad_norm becomes nan, rerun with JP_FP16=0 and lower JP_LR.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_torch",
        max_grad_norm=1.0,
        logging_nan_inf_filter=False,
        predict_with_generate=False,
        remove_unused_columns=True,
    )

    # Newer Transformers versions removed the `tokenizer=` argument from Trainer.
    # Use `processing_class=` instead. This matches the API used by recent Seq2SeqTrainer.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_lora:
        # Merge LoRA weights for simple inference.
        model = trainer.model.merge_and_unload()
    else:
        model = trainer.model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[JP] model saved:", args.output_dir)


if __name__ == "__main__":
    main()
