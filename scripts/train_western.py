import os
import torch
import gc
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

# 언어별 매핑
STRATEGY = {
    'da': 'natural', 'en': 'natural', 'sl': 'natural', 'sr': 'natural', 'iden': 'natural',
    'de': 'sentinel', 'hr': 'natural', 'nl': 'sentinel', 'tr': 'sentinel', 'trde': 'sentinel'
    # it, es 우선 제외
}

class LexNormTransferDataset(Dataset):
    def __init__(self, split="train", target_lang="en", tokenizer_name="google/byt5-small", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.samples = []
        self.fmt = STRATEGY.get(target_lang, 'natural')

        raw_data = load_dataset("weerayut/multilexnorm2026-dev-pub", split=split)
        raw_data = raw_data.filter(lambda x: x['lang'] == target_lang)
        
        for row in raw_data:
            lang = row['lang']
            raw_words = row['raw']
            norm_words = row['norm']
            context_sentence = " ".join(raw_words)

            for i, raw_word in enumerate(raw_words):
                target_word = norm_words[i] if norm_words[i] is not None else raw_word
                
                # natural - 추가 학습 / centinel - 기본 모델 그대로 사용
                if self.fmt == 'natural':
                    input_text = f"lang: {lang} word: {raw_word} context: {context_sentence}"
                else:
                    words_copy = raw_words.copy()
                    words_copy[i] = f"<extra_id_0> {raw_word} <extra_id_1>"
                    input_text = " ".join(words_copy)
                    
                self.samples.append({"input_text": input_text, "target_text": target_word})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        model_inputs = self.tokenizer(sample["input_text"], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.tokenizer(sample["target_text"], max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}

def main():
    target_languages = ['en', 'da', 'de', 'hr', 'nl', 'sl', 'sr', 'tr', 'iden', 'trde']

    for lang in target_languages:
        fmt = STRATEGY.get(lang, 'natural')
        print(f"\n [{lang.upper()}]: {fmt.upper()} 기반 학습")
        
        base_model_name = f"ufal/byt5-small-multilexnorm2021-{lang}"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        except Exception as e:
            continue

        train_data = LexNormTransferDataset(split="train", target_lang=lang, tokenizer_name=base_model_name)
        if len(train_data) == 0: continue

        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear")
        model = get_peft_model(model, peft_config)

        if fmt == 'natural':
            epochs = 1
            lr = 3e-5  # natural 세팅: 과적합 방지 - 많이 테스트해보진 않았습니다. 바꿔보는 것도 좋은 시도
        else:
            epochs = 2
            lr = 3e-5  # 센티넬 세팅: 원본 양식에 2번 적용 - 마찬가지로 많이 테스트 해보진 않았습니다.

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./models/byt5-{lang}", 
            save_strategy="no",             
            learning_rate=lr, 
            per_device_train_batch_size=16, 
            num_train_epochs=epochs, 
            bf16=True, 
            report_to="none"                
        )

        trainer = Seq2SeqTrainer(
            model=model, args=training_args, train_dataset=train_data,
            processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
        )
        trainer.train()

        final_path = f"./final_model/{lang}_model"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        del model, trainer, tokenizer; torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()