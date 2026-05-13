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


STRATEGY = {
    "es": "sentinel",
    "it": "sentinel",
}

UNCHANGED_KEEP_PROB = {
    "es": 0.8,
    "it": 0.8,
}

EPOCHS = {
    "es": 3,
    "it": 2,
}

LEARNING_RATE = {
    "es": 1e-5,
    "it": 1e-5,
}

ES_RULES = {
    "q": ["que", "qué"],
    "qe": ["que", "qué"],
    "ke": ["que", "qué"],
    "k": ["que"],
    "xq": ["porque", "por qué"],
    "xk": ["porque", "por qué"],
    "pq": ["porque", "por qué"],
    "pk": ["porque", "por qué"],
    "pa": ["para"],
    "xa": ["para"],
    "cn": ["con"],
    "tb": ["también"],
    "tmb": ["también"],
    "d": ["de"],
    "aki": ["aquí"],
    "toy": ["estoy"],
    "toi": ["estoy"],
    "kiero": ["quiero"],
}

IT_RULES = {
    "nn": ["non"],
    "nnt": ["niente"],
    "cmq": ["comunque"],
    "qnd": ["quando"],
    "qst": ["questo", "questa", "questi", "queste"],
    "x": ["per"],
    "xke": ["perché"],
    "xké": ["perché"],
    "xchè": ["perché"],
    "ke": ["che"],
    "k": ["che"],
    "sn": ["sono"],
    "dv": ["dove"],
    "tt": ["tutto", "tutti", "tutte"],
    "anke": ["anche"],
    "e'": ["è"],
    "E'": ["È"],
    "perche'": ["perché"],
    "perchè": ["perché"],
    "piu'": ["più"],
    "po": ["po'"],
}


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
    return re.sub(r"(.)\1{" + str(max_repeat) + r",}", lambda m: m.group(1) * max_repeat, s)


def make_key(s: str) -> str:
    s = normalize_apostrophe(s)
    s = s.lower()
    s = strip_accents(s)
    s = collapse_repeats(s, max_repeat=2)
    return s


def build_candidate_dictionary(rows, lang: str):
    """Build candidate dictionary only from training split.

    This is optional. It is used only when --use_candidates_in_prompt is enabled.
    """
    exact = defaultdict(Counter)
    key_index = defaultdict(Counter)

    rules = ES_RULES if lang == "es" else IT_RULES if lang == "it" else {}

    for row in rows:
        raw_words = row["raw"]
        norm_words = row["norm"]

        for raw, norm in zip(raw_words, norm_words):
            target = norm if norm is not None else raw
            exact[raw][target] += 1
            key_index[make_key(raw)][target] += 1

    for raw, norms in rules.items():
        for norm in norms:
            exact[raw][norm] += 1
            key_index[make_key(raw)][norm] += 1

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

    return [norm for norm, _ in merged.most_common(top_k)]


class LexNormTransferDataset(Dataset):
    def __init__(
        self,
        raw_data,
        target_lang: str,
        tokenizer_name: str,
        max_length: int = 128,
        unchanged_keep_prob: float = 0.3,
        seed: int = 42,
        use_candidates_in_prompt: bool = False,
        candidate_dict=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = []
        self.fmt = STRATEGY.get(target_lang, "sentinel")

        rng = random.Random(seed)
        lang_data = raw_data.filter(lambda x, target_lang=target_lang: x["lang"] == target_lang)

        for row in lang_data:
            lang = row["lang"]
            raw_words = row["raw"]
            norm_words = row["norm"]
            context_sentence = " ".join(raw_words)

            for i, raw_word in enumerate(raw_words):
                target_word = norm_words[i] if norm_words[i] is not None else raw_word
                changed = raw_word != target_word

                if not changed and rng.random() > unchanged_keep_prob:
                    continue

                repeat=1
                if changed and target_lang=="es":
                    repeat=3

                if self.fmt == "natural":
                    input_text = f"lang: {lang} word: {raw_word} context: {context_sentence}"
                else:
                    words_copy = list(raw_words)
                    words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                    input_text = " ".join(words_copy)

                if use_candidates_in_prompt:
                    candidates = get_candidates(raw_word, candidate_dict)
                    if candidates:
                        input_text += " candidates: " + " | ".join(candidates)

                for _ in range(repeat):
                    self.samples.append({
                        "input_text": input_text,
                        "target_text": target_word,
                })

        print(f"[{target_lang}] training samples:", len(self.samples))

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
    parser.add_argument("--train_file", type=str, default="./eval_splits/es_it_train.parquet")
    parser.add_argument("--langs", nargs="+", default=["es", "it"])
    parser.add_argument("--output_model_dir", type=str, default="./final_model_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--use_candidates_in_prompt", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1, help="Use a small number like 50 for smoke test.")
    args = parser.parse_args()

    set_seed(args.seed)

    full_dataset = load_dataset(
        "parquet",
        data_files={"train": args.train_file},
    )
    train_split = full_dataset["train"]

    os.makedirs(args.output_model_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    print("CUDA available:", use_cuda)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
        print("bf16:", use_bf16, "fp16:", use_fp16)

    for lang in args.langs:
        fmt = STRATEGY.get(lang, "sentinel")
        print(f"\n[{lang.upper()}] training format = {fmt}")

        base_model_name = f"ufal/byt5-small-multilexnorm2021-{lang}"

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        except Exception as e:
            print(f"[{lang.upper()}] failed to load base model: {base_model_name}")
            print(e)
            continue

        lang_train = train_split.filter(lambda x, lang=lang: x["lang"] == lang)
        if len(lang_train) == 0:
            print(f"[{lang.upper()}] no train data")
            continue

        candidate_dict = None
        if args.use_candidates_in_prompt:
            candidate_dict = build_candidate_dictionary(lang_train, lang)

        train_data = LexNormTransferDataset(
            raw_data=train_split,
            target_lang=lang,
            tokenizer_name=base_model_name,
            max_length=args.max_length,
            unchanged_keep_prob=UNCHANGED_KEEP_PROB.get(lang, 0.3),
            seed=args.seed,
            use_candidates_in_prompt=args.use_candidates_in_prompt,
            candidate_dict=candidate_dict,
        )

        if len(train_data) == 0:
            print(f"[{lang.upper()}] no training samples")
            continue

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./models/byt5-{lang}-eval",
            save_strategy="no",
            learning_rate=LEARNING_RATE.get(lang, 3e-5),
            per_device_train_batch_size=args.batch_size if use_cuda else min(args.batch_size, 2),
            num_train_epochs=EPOCHS.get(lang, 2),
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

        final_path = os.path.join(args.output_model_dir, f"{lang}_model")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

        print(f"[{lang.upper()}] saved to {final_path}")

        del model, trainer, tokenizer, merged_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
