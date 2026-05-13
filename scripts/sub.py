import os
import json
import torch
import gc
import zipfile
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

STRATEGY = {
    'da': 'natural', 'en': 'natural', 'sl': 'natural', 'sr': 'natural', 'hr': 'natural', 'iden': 'natural',
    'de': 'sentinel', 'nl': 'sentinel', 'tr': 'sentinel', 'trde': 'sentinel',
    'es': 'mfr', 'it': 'mfr', # 학습하는거보다 mfr이 더 잘나옴 아직까진
    'id': 'mfr', 'ja': 'mfr', 'ko': 'mfr', 'th': 'mfr', 'vi': 'mfr' # 기존 모델 없는 5개국
}

def build_mfr_dictionary(train_data):
    mfr_counts = {}
    for row in train_data:
        for r, n in zip(row['raw'], row['norm']):
            target = n if n is not None else r
            if r not in mfr_counts: mfr_counts[r] = {}
            mfr_counts[r][target] = mfr_counts[r].get(target, 0) + 1
    return {r: max(targets.items(), key=lambda x: (x[1], x[0] == r))[0] for r, targets in mfr_counts.items()}

def main():
    full_dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train_split, eval_split = full_dataset["train"], full_dataset["test"]
    all_langs = sorted(['en', 'da', 'de', 'es', 'hr', 'it', 'nl', 'sl', 'sr', 'tr', 'iden', 'trde', 'id', 'ja', 'ko', 'th', 'vi'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("./submission_files", exist_ok=True)
    all_predictions_for_json = []

    print("\n추론")

    for lang in all_langs:
        lang_train = train_split.filter(lambda x: x['lang'] == lang)
        lang_eval = eval_split.filter(lambda x: x['lang'] == lang)
        if len(lang_eval) == 0: continue
            
        fmt = STRATEGY.get(lang, 'mfr')
        use_deep_learning = (fmt != 'mfr')
        model_path = f"./final_model/{lang}_model"
        
        tokenizer, model, mfr_dict = None, None, None
        
        if use_deep_learning and os.path.exists(model_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                model.eval()
            except Exception:
                use_deep_learning = False
        else:
            use_deep_learning = False
        
        # 딥러닝을 안 쓰거나 못 쓰는 경우 MFR 사전 구축
        if not use_deep_learning: 
            mfr_dict = build_mfr_dictionary(lang_train)
            print(f"[{lang.upper()}] 예측 (MFR 적용)")

        for row in tqdm(lang_eval, desc=f"[{lang.upper()}] 예측 ", leave=False):
            raw_words = row['raw']
            
            if use_deep_learning:
                inputs_list = []
                context = " ".join(raw_words)
                
                for i, target_word in enumerate(raw_words):
                    if fmt == 'natural':
                        inputs_list.append(f"lang: {lang} word: {target_word} context: {context}")
                    else: # sentinel
                        words_copy = raw_words.copy()
                        words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
                        inputs_list.append(" ".join(words_copy))
                
                inputs = tokenizer(inputs_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                with torch.no_grad(): outputs = model.generate(**inputs, max_length=64, num_beams=2)
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                pred_words = [p.strip() for p in preds]
            else:
                pred_words = [mfr_dict.get(w, w) for w in raw_words]
                
            all_predictions_for_json.append({"raw": raw_words, "pred": pred_words, "lang": lang})

        if use_deep_learning: del model, tokenizer; torch.cuda.empty_cache(); gc.collect()

    json_path = "./submission_files/predictions.json"
    with open(json_path, "w", encoding="utf-8") as f: json.dump(all_predictions_for_json, f, ensure_ascii=False)
    with zipfile.ZipFile("submission.zip", 'w', zipfile.ZIP_DEFLATED) as zipf: zipf.write(json_path, arcname="predictions.json")
    print("\n압축 완료")

if __name__ == "__main__": main()