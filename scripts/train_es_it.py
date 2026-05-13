import os
import re
import gc
import random
import unicodedata
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType

# ============================================================
# es / it 전용 ByT5 학습 코드
# - 기본 전략: sentinel 입력
# - changed token은 전부 사용
# - unchanged token은 일부만 sampling해서 copy-only 학습을 완화
# - candidate dictionary는 옵션으로 prompt에 추가 가능
# ============================================================

DATASET_NAME = os.environ.get("DATASET_NAME", "weerayut/multilexnorm2026-dev-pub")
TARGET_LANGUAGES = ["es", "it"]
STRATEGY = {"es": "sentinel", "it": "sentinel"}

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64
SEED = 42

# 처음에는 False 추천: 기존 ufal ByT5의 sentinel 입력 형식을 최대한 유지
# 후보 dictionary를 입력에 같이 넣어 실험하고 싶으면 train/sub 둘 다 True로 맞추세요.
USE_CANDIDATES_IN_PROMPT = False
TOP_K_CANDIDATES = 5

# unchanged token sampling 비율
# es는 변형/오타를 더 적극적으로 배우게 낮게, it는 정상 token 보존을 위해 조금 높게 둠
UNCHANGED_KEEP_PROB = {
    "es": 0.25,
    "it": 0.45,
}

TRAIN_CONFIG = {
    "es": {"epochs": 3, "lr": 3e-5, "batch_size": 16},
    "it": {"epochs": 2, "lr": 2e-5, "batch_size": 16},
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
    "aqui": ["aquí"],
    "tambien": ["también"],
    "despues": ["después"],
    "kiero": ["quiero"],
    "toy": ["estoy"],
    "toi": ["estoy"],
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
    "perche": ["perché"],
    "piu'": ["più"],
    "piu": ["più"],
    "po": ["po'"],
}


def normalize_apostrophe(s: str) -> str:
    return s.replace("’", "'").replace("`", "'").replace("´", "'")


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def collapse_repeats(s: str, max_repeat: int = 2) -> str:
    # max_repeat=2: buenooo -> buenoo, nn은 보존
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return re.sub(pattern, lambda m: m.group(1) * max_repeat, s)


def make_key(s: str) -> str:
    s = normalize_apostrophe(s)
    s = s.lower()
    s = strip_accents(s)
    s = collapse_repeats(s, max_repeat=2)
    return s


def add_candidate(dic, raw, norm, source, count=1):
    if raw not in dic:
        dic[raw] = {}
    if norm not in dic[raw]:
        dic[raw][norm] = {"count": 0, "sources": set()}
    dic[raw][norm]["count"] += count
    dic[raw][norm]["sources"].add(source)


def finalize_candidates(dic):
    out = {}
    for raw, cand_dict in dic.items():
        total = sum(info["count"] for info in cand_dict.values())
        items = []
        for norm, info in cand_dict.items():
            items.append({
                "norm": norm,
                "count": int(info["count"]),
                "prob": float(info["count"] / total) if total else 0.0,
                "sources": sorted(info["sources"]),
            })
        items.sort(key=lambda x: (x["count"], x["prob"]), reverse=True)
        out[raw] = items
    return out


def build_candidate_dictionary(rows, lang: str):
    exact = {}
    key_index = {}
    rules = ES_RULES if lang == "es" else IT_RULES if lang == "it" else {}

    for row in rows:
        raw_words = row["raw"]
        norm_words = row["norm"]
        for raw, norm in zip(raw_words, norm_words):
            target = norm if norm is not None else raw
            add_candidate(exact, raw, target, "observed", 1)
            add_candidate(key_index, make_key(raw), target, "normalized_key", 1)

    # 언어별 수동 후보. count를 낮게 둬서 observed보다 약한 근거로 둔다.
    for raw, norms in rules.items():
        for norm in norms:
            add_candidate(exact, raw, norm, "rule", 1)
            add_candidate(key_index, make_key(raw), norm, "rule_key", 1)

    return {
        "lang": lang,
        "exact": finalize_candidates(exact),
        "key": finalize_candidates(key_index),
    }


def get_candidates(raw: str, cand_dict: dict, top_k: int = TOP_K_CANDIDATES):
    if cand_dict is None:
        return []

    merged = {}

    def push(c, weight=1.0):
        norm = c["norm"]
        if norm not in merged:
            merged[norm] = {"norm": norm, "score": 0.0, "sources": set()}
        merged[norm]["score"] += float(c.get("count", 1)) * weight
        merged[norm]["sources"].update(c.get("sources", []))

    for c in cand_dict.get("exact", {}).get(raw, []):
        push(c, weight=1.0)

    key = make_key(raw)
    for c in cand_dict.get("key", {}).get(key, []):
        push(c, weight=0.7)

    # copy 후보는 항상 넣되 점수는 낮게 둔다.
    if raw not in merged:
        merged[raw] = {"norm": raw, "score": 0.1, "sources": {"copy"}}

    items = []
    for info in merged.values():
        items.append({
            "norm": info["norm"],
            "score": info["score"],
            "sources": sorted(info["sources"]),
        })
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:top_k]


def format_input(raw_words, i, lang, fmt, cand_dict=None):
    raw_word = raw_words[i]

    if fmt == "natural":
        context = " ".join(raw_words)
        input_text = f"lang: {lang} word: {raw_word} context: {context}"
    else:
        words_copy = list(raw_words)
        words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
        input_text = " ".join(words_copy)

    if USE_CANDIDATES_IN_PROMPT and cand_dict is not None:
        candidates = [c["norm"] for c in get_candidates(raw_word, cand_dict)]
        if candidates:
            input_text = f"{input_text} candidates: {' | '.join(candidates)}"

    return input_text


class LexNormTransferDataset(Dataset):
    def __init__(self, rows, target_lang, tokenizer_name, cand_dict=None, max_length=MAX_INPUT_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = []
        self.fmt = STRATEGY.get(target_lang, "sentinel")
        keep_prob = UNCHANGED_KEEP_PROB.get(target_lang, 0.3)

        for row in rows:
            lang = row["lang"]
            raw_words = row["raw"]
            norm_words = row["norm"]

            for i, raw_word in enumerate(raw_words):
                target_word = norm_words[i] if norm_words[i] is not None else raw_word
                changed = raw_word != target_word

                # changed token은 모두 학습, unchanged token은 일부만 학습
                if not changed and random.random() > keep_prob:
                    continue

                input_text = format_input(raw_words, i, lang, self.fmt, cand_dict)
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
            max_length=MAX_TARGET_LENGTH,
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


def get_precision_flags():
    if not torch.cuda.is_available():
        return False, False
    major, _ = torch.cuda.get_device_capability(0)
    bf16 = major >= 8
    fp16 = not bf16
    return bf16, fp16


def main():
    set_seed(SEED)
    random.seed(SEED)

    full_dataset = load_dataset(DATASET_NAME)
    train_split = full_dataset["train"]

    bf16, fp16 = get_precision_flags()

    for lang in TARGET_LANGUAGES:
        fmt = STRATEGY.get(lang, "sentinel")
        print(f"\n[{lang.upper()}] {fmt.upper()} 기반 ByT5 LoRA 학습")

        base_model_name = f"ufal/byt5-small-multilexnorm2021-{lang}"

        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        except Exception as e:
            print(f"[{lang.upper()}] base model load failed: {base_model_name}")
            print(e)
            print(f"[{lang.upper()}] google/byt5-small로 fallback합니다.")
            base_model_name = "google/byt5-small"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        lang_train = train_split.filter(lambda x: x["lang"] == lang)
        if len(lang_train) == 0:
            print(f"[{lang.upper()}] train data가 없습니다. skip")
            continue

        cand_dict = build_candidate_dictionary(lang_train, lang)
        train_data = LexNormTransferDataset(
            rows=lang_train,
            target_lang=lang,
            tokenizer_name=base_model_name,
            cand_dict=cand_dict,
        )

        print(f"[{lang.upper()}] original sentences: {len(lang_train)}")
        print(f"[{lang.upper()}] training samples after unchanged sampling: {len(train_data)}")

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)

        cfg = TRAIN_CONFIG[lang]
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./models/byt5-{lang}",
            save_strategy="no",
            learning_rate=cfg["lr"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=1,
            num_train_epochs=cfg["epochs"],
            bf16=bf16,
            fp16=fp16,
            logging_steps=50,
            report_to="none",
            remove_unused_columns=False,
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
        print(f"[{lang.upper()}] saved to {final_path}")

        del model, merged_model, trainer, tokenizer, train_data
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
