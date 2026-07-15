import os
import re  
import json
import torch
import gc
import zipfile
import numpy as np
import joblib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

try:
    from ja_rules import build_ja_context, JA_MACRO_MAP
    from ko_rules import build_ko_mfr, KO_MASTER_MAP
except ImportError:
    print("ja_rules.py 또는 ko_rules.py를 찾을 수 없습니다.")
    exit(1)

from es_rules import (is_protected_token, detector_features, generate_candidates, 
                      candidate_features, safe_byt5_output, target_of)

STRATEGY = {
    'da': 'natural', 'en': 'natural', 'sl': 'natural', 'sr': 'natural', 'hr': 'natural', 'iden': 'natural',
    'de': 'sentinel', 'nl': 'sentinel', 'tr': 'sentinel', 'trde': 'sentinel',
    'es': 'es_hybrid',  
    'it': 'mfr',        
    'id': 'mfr', 'th': 'mfr', 'vi': 'mfr', 
    'ko': 'teammate_ko', 
    'ja': 'teammate_ja'  
}

ES_CONFIG = {"detector_threshold": 0.43, "ranker_threshold": 0.25, "low_ranker_byt5_threshold": 0.90}

def build_mfr_with_counts(train_data):
    mfr_dict = {}
    for row in train_data:
        for r, n in zip(row['raw'], row['norm']):
            target = n if n is not None else r
            if r not in mfr_dict: mfr_dict[r] = {}
            mfr_dict[r][target] = mfr_dict[r].get(target, 0) + 1
    return {r: max(t.items(), key=lambda x: (x[1], x[0] == r))[0] for r, t in mfr_dict.items()}

def main():
    epoch = 1
    lr_str = "1e4"
    
    print(f"\n=======================================================")
    print(f"Submission 생성 시작")
    print(f"=======================================================")
    DETECTORS_DIR = "./detectors"
    
    full_dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train_split, eval_split = full_dataset["train"], full_dataset["test"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results_map = [None] * len(eval_split)
    lang_to_indices = {}
    for idx, row in enumerate(eval_split):
        lang_to_indices.setdefault(row['lang'], []).append(idx)

    for lang, indices in lang_to_indices.items():
        lang_train = train_split.filter(lambda x: x['lang'] == lang)
        fmt = STRATEGY.get(lang, 'mfr')
        model_path = "./final_model/es_hybrid_model" if lang == 'es' else f"./comp_model/{lang}_epoch{epoch}_lr{lr_str}"
        
        tokenizer, model = None, None
        es_detector, es_ranker, es_resources = None, None, None
        mfr_dict = build_mfr_with_counts(lang_train)
        ja_context = build_ja_context(lang_train) if lang == 'ja' else None
        ko_mfr_dict = build_ko_mfr(lang_train) if lang == 'ko' else None

        if fmt in ['natural', 'sentinel']:
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                model.eval()
            else: fmt = 'mfr'
        elif fmt == 'es_hybrid':
            det_path, rank_path, res_path = [os.path.join(DETECTORS_DIR, f) for f in ["es_change_detector_rf.joblib", "es_candidate_ranker_rf.joblib", "es_resources.joblib"]]
            if all(os.path.exists(p) for p in [det_path, rank_path, res_path]) and os.path.exists(model_path):
                es_detector = joblib.load(det_path)["model"]
                es_ranker = joblib.load(rank_path)["model"]
                es_resources = joblib.load(res_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                model.eval()
            else: fmt = 'mfr'

        for idx in tqdm(indices, desc=f"Inference {lang}", leave=False):
            row = eval_split[idx]
            raw_words = row['raw']
            pred_words = list(raw_words)
            context = " ".join(raw_words)
            total_len = len(raw_words)

            if fmt == 'teammate_ja':
                for i, w in enumerate(raw_words):
                    prev_w = raw_words[i-1] if i > 0 else "^"
                    next_w = raw_words[i+1] if i < total_len - 1 else "$"
                    if w in JA_MACRO_MAP: pred_words[i] = JA_MACRO_MAP[w]
                    elif i == total_len - 1 and w in ['です', 'ね', 'また']: pred_words[i] = w + ' 。'
                    elif i > 0 and i < total_len - 1 and (prev_w, w, next_w) in ja_context["trigram"]: pred_words[i] = ja_context["trigram"][(prev_w, w, next_w)]
                    elif i < total_len - 1 and (w, next_w) in ja_context["bigram"]: pred_words[i] = ja_context["bigram"][(w, next_w)]
                    elif w in ja_context["unigram"]: pred_words[i] = ja_context["unigram"][w]

            elif fmt == 'teammate_ko':
                for i, w in enumerate(raw_words):
                    pred_words[i] = KO_MASTER_MAP[w] if w in KO_MASTER_MAP else ko_mfr_dict.get(w, w)

            elif fmt in ['natural', 'sentinel'] and model is not None:
                for i, tw in enumerate(raw_words):
                    inp = f"lang: {lang} word: {tw} context: {context}" if fmt == 'natural' else " ".join([f"<extra_id_0> {tw} <extra_id_1>" if j==i else w for j, w in enumerate(raw_words)])
                    ins = tokenizer(inp, return_tensors="pt", truncation=True, max_length=128).to(device)
                    with torch.no_grad(): out = model.generate(**ins, max_length=64, num_beams=2)
                    pred_words[i] = tokenizer.decode(out[0], skip_special_tokens=True).strip()

            elif fmt == 'es_hybrid':
                for i, raw in enumerate(raw_words):
                    if is_protected_token(raw): continue
                    left = raw_words[i - 1] if i > 0 else "<BOS>"
                    right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
                    if es_detector.predict_proba([detector_features(raw, left, right, es_resources)])[0][1] < ES_CONFIG["detector_threshold"]: continue 
                    cands = generate_candidates(raw, mfr=es_resources["mfr"], mfr_conf=es_resources["mfr_conf"], key_map=es_resources["key_map"])
                    feats = [candidate_features(raw, cand, source, left, right, es_resources) for source, cand in cands]
                    probs = es_ranker.predict_proba(feats)[:, 1]
                    best_i = int(np.argmax(probs))
                    if probs[best_i] >= ES_CONFIG["ranker_threshold"] and cands[best_i][1] != raw: pred_words[i] = cands[best_i][1]
                    elif cands[best_i][1] == raw or probs[best_i] < ES_CONFIG["low_ranker_byt5_threshold"]:
                        tmp = raw_words.copy(); tmp[i] = f"<extra_id_0> {raw} <extra_id_1>"
                        ins = tokenizer(" ".join(tmp), return_tensors="pt", truncation=True, max_length=128).to(device)
                        with torch.no_grad(): out = model.generate(**ins, max_new_tokens=12, num_beams=1, repetition_penalty=1.2, no_repeat_ngram_size=3)
                        pred_words[i] = safe_byt5_output(raw, tokenizer.decode(out[0], skip_special_tokens=True).strip(), allow_underscore=True) or raw
            else:
                pred_words = [mfr_dict.get(w, w) for w in raw_words]
            
            results_map[idx] = {"raw": raw_words, "pred": pred_words, "lang": lang}

        if model is not None: 
            model.cpu()
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    os.makedirs("./submission_files", exist_ok=True)
    json_path = "./submission_files/predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_map, f, ensure_ascii=False)
    zip_filename = "submission.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")
    print(f"submission 생성 완료")

if __name__ == "__main__": main()