import os
import torch
import gc
import shutil
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
#@@@@@@@@@@@@@@@
#현재까지 최신본
#@@@@@@@@@@@@@@@@@@
STRATEGY = {
    'da': 'natural', 'en': 'natural', 'sl': 'natural', 'sr': 'natural', 'iden': 'natural',
    'de': 'sentinel', 'hr': 'natural', 'nl': 'sentinel', 'tr': 'sentinel', 'trde': 'sentinel'
}

class CheckpointTrackerCallback(TrainerCallback):
    def __init__(self):
        self.epoch_checkpoints = {}

    def on_save(self, args, state, control, **kwargs):
        current_epoch = int(round(state.epoch))
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.epoch_checkpoints[current_epoch] = ckpt_path

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
    
    param_grid = [
        {"lr": 1e-4, "lr_str": "1e4", "max_epochs": 1, "target_epochs": [1]}
    ]

    os.makedirs("./comp_model", exist_ok=True)

    for params in param_grid:
        max_epochs = params["max_epochs"]
        target_epochs = params["target_epochs"]
        lr = params["lr"]
        lr_str = params["lr_str"]
        
        print(f"\n" + "="*60)
        print(f"총 Epoch: {max_epochs} | LR: {lr_str} | 타겟: {target_epochs}")
        print("="*60)

        for lang in target_languages:
            fmt = STRATEGY.get(lang, 'natural')
            base_model_name = f"ufal/byt5-small-multilexnorm2021-{lang}"
            tmp_model_dir = f"./models/tmp_{lang}"
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            except Exception as e:
                print(f"[{lang}] 베이스 모델 로드 실패. 스킵합니다.")
                continue

            train_data = LexNormTransferDataset(split="train", target_lang=lang, tokenizer_name=base_model_name)
            if len(train_data) == 0: continue

            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear")
            model = get_peft_model(model, peft_config)

            tracker = CheckpointTrackerCallback()

            training_args = Seq2SeqTrainingArguments(
                output_dir=tmp_model_dir, 
                save_strategy="epoch",
                learning_rate=lr, 
                per_device_train_batch_size=16, 
                num_train_epochs=max_epochs,
                bf16=True if torch.cuda.is_available() else False, 
                report_to="none"                
            )

            trainer = Seq2SeqTrainer(
                model=model, args=training_args, train_dataset=train_data,
                processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                callbacks=[tracker]               
            )
            print(f"\n - [{lang.upper()}] 1~{max_epochs} 에포크 연속 훈련 시작...")
            trainer.train()

            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()


            print(f" - [{lang.upper()}] 훈련 완료.")
            
            for ep in target_epochs:
                ckpt_dir = tracker.epoch_checkpoints.get(ep)
                if not ckpt_dir:
                    print(f"에포크 {ep} 의 체크포인트를 찾을 수 없습니다.")
                    continue
                
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                
                peft_model = PeftModel.from_pretrained(base_model, ckpt_dir)
                
                merged_model = peft_model.merge_and_unload()
                
                final_path = f"./comp_model/{lang}_epoch{ep}_lr{lr_str}"
                merged_model.save_pretrained(final_path)
                tokenizer.save_pretrained(final_path)
                
                print(f"추출 완료: {final_path}")
                
                del base_model, peft_model, merged_model
                torch.cuda.empty_cache()
                gc.collect()

            if os.path.exists(tmp_model_dir):
                shutil.rmtree(tmp_model_dir)

if __name__ == "__main__":
    main()