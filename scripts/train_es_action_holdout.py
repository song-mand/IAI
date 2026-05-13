import argparse
import gc
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType


LANG = "es"
BASE_MODEL_NAME = "ufal/byt5-small-multilexnorm2021-es"

COPY_LABEL = "COPY"
NORM_PREFIX = "NORM"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_accents(s: str) -> str:
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


def build_observed_candidate_dictionary(rows):
    """Build candidates automatically from train data only.

    No hand-written ES rules are used.
    Candidate sources:
      1. exact raw -> observed norm
      2. normalized-key(raw) -> observed norm
    """
    exact = defaultdict(Counter)
    key_index = defaultdict(Counter)

    for row in rows:
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


class ActionAwareESDataset(Dataset):
    """Spanish-only action-aware lexical normalization dataset.

    Previous target format:
      unchanged: target = raw
      changed:   target = norm

    New target format:
      unchanged: target = <COPY>
      changed:   target = <NORM> norm
    """

    def __init__(
        self,
        raw_data,
        tokenizer_name: str,
        max_length: int = 128,
        copy_keep_prob: float = 0.8,
        changed_repeat: int = 3,
        seed: int = 42,
        use_candidates_in_prompt: bool = False,
        candidate_dict=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = []

        rng = random.Random(seed)
        lang_data = raw_data.filter(lambda x: x["lang"] == LANG)

        copy_count = 0
        norm_count = 0

        for row in lang_data:
            raw_words = row["raw"]
            norm_words = row["norm"]

            for i, raw_word in enumerate(raw_words):
                target_word = norm_words[i] if norm_words[i] is not None else raw_word
                changed = raw_word != target_word

                if not changed and rng.random() > copy_keep_prob:
                    continue

                words_copy = list(raw_words)
                words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                marked_sentence = " ".join(words_copy)
                """
                input_text = (
                    f"task: lexical_normalization "
                    f"lang: {LANG} "
                    f"target: {raw_word} "
                    f"sentence: {marked_sentence}"
                )
                """
                input_text=marked_sentence

                if use_candidates_in_prompt:
                    candidates = get_candidates(raw_word, candidate_dict)
                    if candidates:
                        input_text += " candidates: " + " | ".join(candidates)

                if changed:
                    target_text = f"{NORM_PREFIX} {target_word}"
                    repeat = max(1, changed_repeat)
                    norm_count += repeat
                else:
                    target_text = COPY_LABEL
                    repeat = 1
                    copy_count += 1

                for _ in range(repeat):
                    self.samples.append({
                        "input_text": input_text,
                        "target_text": target_text,
                    })

        print(f"[ES] samples: {len(self.samples)}")
        print(f"[ES] COPY-label samples: {copy_count}")
        print(f"[ES] NORM-label samples after repeat: {norm_count}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./eval_splits_es/es_train.parquet")
    parser.add_argument("--output_model_dir", type=str, default="./final_model_eval_es_action")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--copy_keep_prob", type=float, default=0.8)
    parser.add_argument("--changed_repeat", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_candidates_in_prompt", action="store_true")
    parser.add_argument("--lora_target", choices=["qv", "all-linear"], default="qv")
    args = parser.parse_args()

    set_seed(args.seed)

    full_dataset = load_dataset(
        "parquet",
        data_files={"train": args.train_file},
    )
    train_split = full_dataset["train"]
    es_train = train_split.filter(lambda x: x["lang"] == LANG)

    if len(es_train) == 0:
        raise ValueError(f"No ES data found in {args.train_file}")

    os.makedirs(args.output_model_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    print("CUDA available:", use_cuda)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
        print("bf16:", use_bf16, "fp16:", use_fp16)

    print("[ES] base model:", BASE_MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    model.config.tie_word_embeddings = False

    candidate_dict = None
    if args.use_candidates_in_prompt:
        candidate_dict = build_observed_candidate_dictionary(es_train)
        print("[ES] observed candidate dictionary enabled")

    train_data = ActionAwareESDataset(
        raw_data=train_split,
        tokenizer_name=BASE_MODEL_NAME,
        max_length=args.max_length,
        copy_keep_prob=args.copy_keep_prob,
        changed_repeat=args.changed_repeat,
        seed=args.seed,
        use_candidates_in_prompt=args.use_candidates_in_prompt,
        candidate_dict=candidate_dict,
    )

    if args.lora_target == "qv":
        target_modules = ["q", "v"]
        r = 8
        alpha = 16
    else:
        target_modules = "all-linear"
        r = 16
        alpha = 32

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/byt5-es-action-eval",
        save_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size if use_cuda else min(args.batch_size, 2),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        logging_steps=20,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()

    final_path = os.path.join(args.output_model_dir, "es_model")
    merged_model = trainer.model.merge_and_unload()
    merged_model.config.tie_word_embeddings = False
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"[ES] saved to {final_path}")

    del model, trainer, tokenizer, merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
