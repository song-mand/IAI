import os
import re
import gc
import json
import zipfile
import unicodedata
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# 추론 코드
# - es/it은 학습된 ByT5 모델 사용
# - dictionary는 MFR single-best가 아니라 candidate dictionary로 생성
# - 모델 출력이 비어 있거나 너무 이상할 때 candidate dictionary로 fallback
# - 출력 순서는 원본 test split 순서를 유지
# ============================================================

DATASET_NAME = os.environ.get("DATASET_NAME", "weerayut/multilexnorm2026-dev-pub")

STRATEGY = {
    "da": "natural", "en": "natural", "sl": "natural", "sr": "natural", "hr": "natural", "iden": "natural",
    "de": "sentinel", "nl": "sentinel", "tr": "sentinel", "trde": "sentinel",
    "es": "sentinel", "it": "sentinel",
    "id": "mfr", "ja": "mfr", "ko": "mfr", "th": "mfr", "vi": "mfr",
}

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64
TOP_K_CANDIDATES = 5
USE_CANDIDATES_IN_PROMPT = False  # train_es_it.py와 반드시 같은 값으로 맞추세요.

# model_first: 모델 예측을 우선 사용. 이상한 출력일 때만 후보 사전 fallback.
# dict_when_copy: 모델이 raw를 그대로 복사했고, 사전에 강한 변경 후보가 있으면 후보로 바꿈.
HYBRID_MODE = "model_first"

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


def build_candidate_dictionary(train_data, lang: str):
    exact = {}
    key_index = {}
    rules = ES_RULES if lang == "es" else IT_RULES if lang == "it" else {}

    for row in train_data:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            add_candidate(exact, raw, target, "observed", 1)
            add_candidate(key_index, make_key(raw), target, "normalized_key", 1)

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
        push(c, 1.0)

    key = make_key(raw)
    for c in cand_dict.get("key", {}).get(key, []):
        push(c, 0.7)

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


def best_candidate(raw: str, cand_dict: dict):
    candidates = get_candidates(raw, cand_dict, top_k=1)
    return candidates[0]["norm"] if candidates else raw


def build_mfr_dictionary(train_data):
    # 기존 언어용 fallback. es/it에는 기본적으로 쓰지 않음.
    mfr_counts = {}
    for row in train_data:
        for r, n in zip(row["raw"], row["norm"]):
            target = n if n is not None else r
            if r not in mfr_counts:
                mfr_counts[r] = {}
            mfr_counts[r][target] = mfr_counts[r].get(target, 0) + 1
    return {r: max(targets.items(), key=lambda x: (x[1], x[0] == r))[0] for r, targets in mfr_counts.items()}


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


def suspicious_prediction(raw: str, pred: str) -> bool:
    pred = pred.strip()
    if pred == "":
        return True
    # 단일 token normalization인데 너무 긴 문자열이면 위험하게 봄.
    if len(pred) > max(30, len(raw) * 4):
        return True
    # T5 sentinel이 그대로 남으면 실패로 봄.
    if "<extra_id_" in pred:
        return True
    return False


def apply_hybrid(raw: str, pred: str, cand_dict: dict) -> str:
    pred = pred.strip()

    if suspicious_prediction(raw, pred):
        return best_candidate(raw, cand_dict)

    if HYBRID_MODE == "dict_when_copy" and pred == raw:
        candidates = get_candidates(raw, cand_dict, top_k=2)
        if candidates:
            top = candidates[0]
            # copy가 아닌 후보가 압도적으로 강할 때만 바꿈.
            if top["norm"] != raw and top["score"] >= 2.0:
                return top["norm"]

    return pred


def main():
    full_dataset = load_dataset(DATASET_NAME)
    train_split, eval_split = full_dataset["train"], full_dataset["test"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("./submission_files", exist_ok=True)
    predictions_by_idx = [None] * len(eval_split)

    all_langs = sorted(set(eval_split["lang"]))
    print("\n추론")
    print(f"device: {device}")

    for lang in all_langs:
        eval_items = [(idx, row) for idx, row in enumerate(eval_split) if row["lang"] == lang]
        if not eval_items:
            continue

        lang_train = train_split.filter(lambda x: x["lang"] == lang)
        fmt = STRATEGY.get(lang, "mfr")
        model_path = f"./final_model/{lang}_model"

        use_deep_learning = fmt != "mfr" and os.path.exists(model_path)
        tokenizer, model = None, None

        cand_dict = None
        if lang in {"es", "it"}:
            cand_dict = build_candidate_dictionary(lang_train, lang)

        if use_deep_learning:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                model.eval()
                print(f"[{lang.upper()}] 예측 (MODEL: {fmt}, path={model_path})")
            except Exception as e:
                print(f"[{lang.upper()}] model load failed: {e}")
                use_deep_learning = False

        if not use_deep_learning:
            print(f"[{lang.upper()}] 예측 (dictionary/MFR fallback)")
            mfr_dict = build_mfr_dictionary(lang_train)
        else:
            mfr_dict = None

        for idx, row in tqdm(eval_items, desc=f"[{lang.upper()}] 예측", leave=False):
            raw_words = row["raw"]

            if use_deep_learning:
                inputs_list = [
                    format_input(raw_words, i, lang, fmt, cand_dict)
                    for i in range(len(raw_words))
                ]
                inputs = tokenizer(
                    inputs_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_INPUT_LENGTH,
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=MAX_TARGET_LENGTH,
                        num_beams=2,
                    )

                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                pred_words = [apply_hybrid(r, p, cand_dict) for r, p in zip(raw_words, preds)]
            else:
                if lang in {"es", "it"} and cand_dict is not None:
                    pred_words = [best_candidate(w, cand_dict) for w in raw_words]
                else:
                    pred_words = [mfr_dict.get(w, w) for w in raw_words]

            predictions_by_idx[idx] = {
                "raw": raw_words,
                "pred": pred_words,
                "lang": lang,
            }

        if use_deep_learning:
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    # 혹시 빠진 항목이 있으면 안전하게 copy로 채움.
    for idx, item in enumerate(predictions_by_idx):
        if item is None:
            row = eval_split[idx]
            predictions_by_idx[idx] = {
                "raw": row["raw"],
                "pred": row["raw"],
                "lang": row["lang"],
            }

    json_path = "./submission_files/predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions_by_idx, f, ensure_ascii=False)

    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")

    print("\n압축 완료: submission.zip")


if __name__ == "__main__":
    main()
