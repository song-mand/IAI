import os
import gc
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

"""
train_es_it_v3.py

V3 training file.

Purpose:
- Train ByT5-based models only for Spanish (es) and Italian (it).
- Korean is intentionally NOT trained in v3.
- Korean is handled conservatively in sub_v3_es_it_ko.py with MFR + safe rules.

Saved models:
- ./final_model/es_model
- ./final_model/it_model
"""

TARGET_LANGUAGES = ["es", "it"]

PROMPT_TYPE = {
    "es": "marked_natural",
    "it": "marked_natural",
}


class LexNormDataset(Dataset):
    def __init__(
        self,
        split="train",
        target_lang="es",
        tokenizer_name="google/byt5-small",
        max_length=128,
        prompt_type="marked_natural",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = []
        self.prompt_type = prompt_type

        raw_data = load_dataset("weerayut/multilexnorm2026-dev-pub", split=split)
        raw_data = raw_data.filter(lambda x: x["lang"] == target_lang)

        for row in raw_data:
            lang = row["lang"]
            raw_words = row["raw"]
            norm_words = row["norm"]
            plain_context = " ".join(raw_words)

            for i, raw_word in enumerate(raw_words):
                target_word = norm_words[i] if norm_words[i] is not None else raw_word

                words_copy = raw_words.copy()
                words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                marked_context = " ".join(words_copy)

                if prompt_type == "marked_natural":
                    input_text = (
                        f"normalize lang: {lang} "
                        f"target: {raw_word} "
                        f"context: {marked_context}"
                    )
                elif prompt_type == "natural":
                    input_text = (
                        f"lang: {lang} "
                        f"word: {raw_word} "
                        f"context: {plain_context}"
                    )
                elif prompt_type == "sentinel":
                    input_text = marked_context
                else:
                    input_text = raw_word

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


def get_precision_flags():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    elif torch.cuda.is_available():
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def load_base_model(lang):
    candidates = [
        f"ufal/byt5-small-multilexnorm2021-{lang}",
        "google/byt5-small",
    ]

    last_error = None
    for model_name in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"[{lang.upper()}] Loaded base model: {model_name}")
            return tokenizer, model, model_name
        except Exception as e:
            print(f"[{lang.upper()}] Failed to load base model: {model_name}")
            last_error = e

    raise RuntimeError(f"Could not load model for {lang}: {last_error}")


def build_lora_model(model):
    try:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules="all-linear",
        )
        return get_peft_model(model, peft_config)
    except Exception as e:
        print("[LoRA] target_modules='all-linear' failed. Fallback to ['q', 'v'].")
        print(e)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"],
        )
        return get_peft_model(model, peft_config)


def get_train_setting(lang):
    return {"epochs": 2, "lr": 3e-5, "batch_size": 16}


def train_one_language(lang):
    print(f"\n========== Training {lang.upper()} ==========")
    tokenizer, model, base_model_name = load_base_model(lang)

    train_data = LexNormDataset(
        split="train",
        target_lang=lang,
        tokenizer_name=base_model_name,
        max_length=128,
        prompt_type=PROMPT_TYPE.get(lang, "marked_natural"),
    )

    if len(train_data) == 0:
        print(f"[{lang.upper()}] No train samples. Skipped.")
        return

    model = build_lora_model(model)
    precision_flags = get_precision_flags()
    setting = get_train_setting(lang)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./models/byt5-{lang}-v3",
        save_strategy="no",
        learning_rate=setting["lr"],
        per_device_train_batch_size=setting["batch_size"],
        num_train_epochs=setting["epochs"],
        logging_steps=100,
        report_to="none",
        **precision_flags,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()

    final_path = f"./final_model/{lang}_model"
    os.makedirs(final_path, exist_ok=True)

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"[{lang.upper()}] Saved model to {final_path}")

    del model, trainer, tokenizer, merged_model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    os.makedirs("./final_model", exist_ok=True)
    for lang in TARGET_LANGUAGES:
        train_one_language(lang)


if __name__ == "__main__":
    main()
