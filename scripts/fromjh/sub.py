import os
import re  
import json
import torch
import gc
import zipfile
import numpy as np
import joblib
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
"""
try:
    from ja_rules import build_ja_context, JA_MACRO_MAP
    from ko_rules import build_ko_mfr, KO_MASTER_MAP
except ImportError:
    print("ja_rules.py 또는 ko_rules.py를 찾을 수 없습니다.")
    exit(1)
    """

from es_rules import (is_protected_token, detector_features, generate_candidates, 
                      candidate_features, safe_byt5_output, target_of)

STRATEGY = {
    'da': 'natural', 'en': 'natural', 'sl': 'natural', 'sr': 'natural', 'hr': 'natural', 'iden': 'natural',
    'de': 'sentinel', 'nl': 'sentinel', 'tr': 'sentinel', 'trde': 'sentinel',
    'es': 'es_hybrid',  
    'it': 'it_hybrid_rf_byt5',        
    'id': 'mfr', 'th': 'mfr', 'vi': 'mfr', 
    #'ko': 'ko_hybrid',  
    'ko': 'mfr', 
    #'ja': 'cmfr_hybrid'  
    'ja': 'mfr'  
}

ES_CONFIG = {"detector_threshold": 0.43, "ranker_threshold": 0.25, "low_ranker_byt5_threshold": 0.90}

IT_CONFIG = {
    "detector_path": "./detectors/it_change_detector_rf.joblib",
    "model_path": "./final_model/it_model",
    "threshold": 0.65,
    "mfr_min_conf": 0.70,
}

def build_mfr_with_counts(train_data):
    mfr_dict = {}
    for row in train_data:
        for r, n in zip(row['raw'], row['norm']):
            target = n if n is not None else r
            if r not in mfr_dict: mfr_dict[r] = {}
            mfr_dict[r][target] = mfr_dict[r].get(target, 0) + 1
    return {r: max(t.items(), key=lambda x: (x[1], x[0] == r))[0] for r, t in mfr_dict.items()}

####################################..
def build_mfr_with_conf(train_data):
    counts = defaultdict(Counter)

    for row in train_data:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1

    mfr = {}
    conf = {}

    for raw, counter in counts.items():
        total = sum(counter.values())
        best, best_count = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
        mfr[raw] = best
        conf[raw] = best_count / total if total else 0.0

    return mfr, conf

def it_strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def it_normalize_apostrophe(s: str) -> str:
    return (
        s.replace("’", "'")
         .replace("`", "'")
         .replace("´", "'")
         .replace("‘", "'")
    )


def it_collapse_repeats(s: str, max_repeat: int = 2) -> str:
    return re.sub(
        r"(.)\1{" + str(max_repeat) + r",}",
        lambda m: m.group(1) * max_repeat,
        s,
    )


def it_make_key(s: str) -> str:
    s = it_normalize_apostrophe(s)
    s = s.lower()
    s = it_strip_accents(s)
    s = it_collapse_repeats(s, max_repeat=2)
    return s


def it_is_protected_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return True
    if t.startswith("@") or t.startswith("#"):
        return True
    if t.startswith("http://") or t.startswith("https://"):
        return True
    if re.fullmatch(r"\d+([.,:/-]\d+)*", t):
        return True
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True
    return False


def it_has_long_repetition(token: str) -> bool:
    return re.search(r"(.)\1{2,}", token.lower()) is not None


def it_has_laughter_pattern(token: str) -> bool:
    t = token.lower()
    return bool(re.search(r"(ja){2,}", t) or re.search(r"(jaja){1,}", t))


def it_token_shape(token: str) -> str:
    out = []
    for ch in token:
        base = it_strip_accents(ch)
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("0")
        elif base != ch:
            out.append("á")
        else:
            out.append(ch)
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(out))


def it_get_raw_stat_features(raw: str, raw_stats):
    s = raw_stats.get(raw)
    if s is None:
        return {
            "raw_seen": 0,
            "raw_total": 0,
            "raw_change_prob": 0.0,
            "raw_copy_prob": 0.0,
            "raw_best_is_copy": 0,
            "raw_best_prob": 0.0,
        }

    return {
        "raw_seen": 1,
        "raw_total": min(s["total"], 10),
        "raw_change_prob": s["change_prob"],
        "raw_copy_prob": s["copy_prob"],
        "raw_best_is_copy": int(s["best_norm"] == raw),
        "raw_best_prob": s["best_prob"],
    }


def it_get_key_stat_features(raw: str, key_stats):
    key = it_make_key(raw)
    s = key_stats.get(key)
    if s is None:
        return {
            "key_seen": 0,
            "key_total": 0,
            "key_best_prob": 0.0,
            "key_best_is_raw": 0,
        }

    return {
        "key_seen": 1,
        "key_total": min(s["total"], 10),
        "key_best_prob": s["best_prob"],
        "key_best_is_raw": int(s["best_norm"] == raw),
    }


def it_detector_features(raw: str, left: str, right: str, raw_stats, key_stats):
    t = raw
    low = t.lower()
    key = it_make_key(t)
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    punct = sum((not ch.isalnum()) for ch in t)

    feats = {
        "bias": 1,
        "raw_lower=" + low: 1,
        "key=" + key: 1,
        "shape=" + it_token_shape(t): 1,
        "left_lower=" + left.lower(): 1,
        "right_lower=" + right.lower(): 1,
        "prefix1=" + low[:1]: 1,
        "prefix2=" + low[:2]: 1,
        "prefix3=" + low[:3]: 1,
        "suffix1=" + low[-1:]: 1,
        "suffix2=" + low[-2:]: 1,
        "suffix3=" + low[-3:]: 1,
        "len": min(len(t), 30),
        "letters": min(letters, 30),
        "digits": min(digits, 30),
        "punct": min(punct, 30),
        "is_protected": int(it_is_protected_token(t)),
        "starts_at": int(t.startswith("@")),
        "starts_hash": int(t.startswith("#")),
        "starts_http": int(t.startswith("http://") or t.startswith("https://")),
        "is_digit_like": int(bool(re.fullmatch(r"\d+([.,:/-]\d+)*", t))),
        "has_long_repetition": int(it_has_long_repetition(t)),
        "has_laughter_pattern": int(it_has_laughter_pattern(t)),
        "has_accent": int(it_strip_accents(t) != t),
        "is_all_lower": int(t.islower()),
        "is_all_upper": int(t.isupper()),
        "is_title": int(t[:1].isupper() and t[1:].islower()),
        "has_qkx": int(any(ch in low for ch in ["q", "k", "x"])),
        "has_underscore": int("_" in t),
        "has_apostrophe": int("'" in it_normalize_apostrophe(t)),
    }

    feats.update(it_get_raw_stat_features(raw, raw_stats))
    feats.update(it_get_key_stat_features(raw, key_stats))
    return feats


def it_safe_generation(raw: str, pred: str) -> str:
    pred = (pred or "").strip()

    if not pred:
        return raw
    if pred in {"<pad>", "</s>", "<s>"}:
        return raw
    if len(pred) > max(40, len(raw) * 4 + 10):
        return raw
    if "<extra_id_" in pred:
        return raw

    return pred


def predict_it_hybrid_row(raw_words, resources, device):
    detector = resources["detector"]
    detector_model = detector["model"]
    raw_stats = detector["raw_stats"]
    key_stats = detector["key_stats"]

    threshold = resources["threshold"]
    mfr_min_conf = resources["mfr_min_conf"]
    mfr = resources["mfr"]
    mfr_conf = resources["mfr_conf"]
    tokenizer = resources.get("tokenizer")
    model = resources.get("model")

    pred_words = list(raw_words)
    byt5_inputs = []
    byt5_positions = []
    context = " ".join(raw_words)

    for i, raw in enumerate(raw_words):
        if it_is_protected_token(raw):
            pred_words[i] = raw
            continue

        # 1차: MFR이 충분히 확실하면 MFR 사용
        mfr_pred = mfr.get(raw, raw)
        conf = mfr_conf.get(raw, 0.0)
        if mfr_pred != raw and conf >= mfr_min_conf:
            pred_words[i] = mfr_pred
            continue

        # 2차: RF detector가 바꿔야 한다고 판단할 때만 ByT5 사용
        left = raw_words[i - 1] if i > 0 else "<BOS>"
        right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

        feats = it_detector_features(raw, left, right, raw_stats, key_stats)
        prob_change = detector_model.predict_proba([feats])[0][1]

        if prob_change < threshold or tokenizer is None or model is None:
            pred_words[i] = raw
            continue

        byt5_inputs.append(f"lang: it word: {raw} context: {context}")
        byt5_positions.append(i)

    if byt5_inputs:
        inputs = tokenizer(
            byt5_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=2)

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pos, raw, pred in zip(
            byt5_positions,
            [raw_words[p] for p in byt5_positions],
            preds
        ):
            pred_words[pos] = it_safe_generation(raw, pred)

    return pred_words
#####################################..

def train_cmfr_classifier(train_data):
    X_train_text, y_train = [], []
    for row in train_data:
        raw_words, norm_words = row['raw'], row['norm']
        for i, (r, n) in enumerate(zip(raw_words, norm_words)):
            target = n if n is not None else r
            prev_w = raw_words[i-1] if i > 0 else ""
            next_w = raw_words[i+1] if i < len(raw_words)-1 else ""
            X_train_text.append(f"{prev_w} {r} {next_w}")
            y_train.append(1 if r != target else 0)
    vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_features=10000)
    X = vec.fit_transform(X_train_text)
    clf = LogisticRegression(max_iter=1000).fit(X, y_train)
    return vec, clf

def main():
    epoch = 6
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
        model_path = "./final_model/ja_model" if lang == 'ja' else ("./final_model/es_hybrid_model" if lang == 'es' else f"./comp_model/{lang}_epoch{epoch}_lr{lr_str}")
        
        tokenizer, model = None, None
        vectorizer, classifier, mfr_counts = None, None, None
        es_detector, es_ranker, es_resources = None, None, None
        ja_context, ko_mfr_dict = None, None
        #######
        it_detector, it_resources = None, None
        ########
        if fmt == 'cmfr_hybrid':
            ja_context = build_ja_context(lang_train)
            vectorizer, classifier = train_cmfr_classifier(lang_train)
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                tokenizer.padding_side = "left"
                if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
                model.eval()
            else: fmt = 'mfr'
        elif fmt == 'ko_hybrid':
            ko_mfr_dict = build_ko_mfr(lang_train)
        elif fmt in ['natural', 'sentinel']:
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
        ###########################################    
        elif fmt == 'it_hybrid_rf_byt5':
            config = IT_CONFIG
            mfr_dict, mfr_conf = build_mfr_with_conf(lang_train)

            if os.path.exists(config["detector_path"]):
                it_detector = joblib.load(config["detector_path"])
            else:
                print(f"[IT] detector missing -> fallback MFR: {config['detector_path']}")
                fmt = 'mfr'
                mfr_counts = mfr_dict

            if fmt == 'it_hybrid_rf_byt5':
                if os.path.exists(config["model_path"]):
                    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
                    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_path"]).to(device)
                    model.eval()
                else:
                    print(f"[IT] ByT5 model missing -> detector/MFR only: {config['model_path']}")
                    tokenizer, model = None, None

                it_resources = {
                    "detector": it_detector,
                    "threshold": config["threshold"],
                    "mfr_min_conf": config["mfr_min_conf"],
                    "mfr": mfr_dict,
                    "mfr_conf": mfr_conf,
                    "tokenizer": tokenizer,
                    "model": model,
                }
        #####################################################
        if fmt == 'mfr':
            mfr_counts = build_mfr_with_counts(lang_train)

        for idx in tqdm(indices, desc=f"Inference {lang}", leave=False):
            row = eval_split[idx]
            raw_words = row['raw']
            pred_words = list(raw_words)
            context = " ".join(raw_words)
            total_len = len(raw_words)

            if fmt == 'cmfr_hybrid' and model is not None:
                for i, target_word in enumerate(raw_words):
                    prev_w = raw_words[i-1] if i > 0 else "^"
                    next_w = raw_words[i+1] if i < total_len - 1 else "$"
                    pred_ngram = None
                    if target_word in JA_MACRO_MAP: pred_ngram = JA_MACRO_MAP[target_word]
                    elif i == total_len - 1 and target_word in ['です', 'ね', 'また']: pred_ngram = target_word + ' 。'
                    elif i > 0 and i < total_len - 1 and (prev_w, target_word, next_w) in ja_context["trigram"]: pred_ngram = ja_context["trigram"][(prev_w, target_word, next_w)]
                    elif i < total_len - 1 and (target_word, next_w) in ja_context["bigram"]: pred_ngram = ja_context["bigram"][(target_word, next_w)]
                    elif target_word in ja_context["unigram"]: pred_ngram = ja_context["unigram"][target_word]
                    
                    if pred_ngram is not None:
                        pred_words[i] = pred_ngram
                    else:
                        prob = classifier.predict_proba(vectorizer.transform([f"{prev_w} {target_word} {next_w}"]))[0][1]
                        if prob < 0.70: continue
                        prompt = f"文脈: {context}\n単語: {target_word}\n訂正:"
                        ins = tokenizer(prompt, return_tensors="pt").to(device)
                        with torch.no_grad(): out = model.generate(**ins, max_new_tokens=len(target_word)+5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                        gen = tokenizer.decode(out[0][ins.input_ids.shape[1]:], skip_special_tokens=True).split('\n')[0].strip()
                        if not gen or len(gen) > len(target_word) + 3 or (len(target_word) >= 2 and len(set(gen) & set(target_word)) == 0): pred_words[i] = target_word
                        else: pred_words[i] = gen

            elif fmt == 'ko_hybrid':
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
            ###########################
            elif fmt == 'it_hybrid_rf_byt5':
                pred_words = predict_it_hybrid_row(raw_words, it_resources, device)
            ###########################
            
            else:
                pred_words = [mfr_counts.get(w, w) for w in raw_words]
            
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
    zip_filename = "submission_new3.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")
    print(f"submission_hybrid.zip 생성 완료")

if __name__ == "__main__": main()