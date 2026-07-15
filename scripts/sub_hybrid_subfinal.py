import gc
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import joblib
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Western languages follow train_western.py.
STRATEGY = {
    "da": "mfr",
    "en": "mfr",
    "sl": "mfr",
    "sr": "mfr",
    "hr": "mfr",
    "iden": "mfr",
    "de": "mfr",
    "nl": "mfr",
    "tr": "mfr",
    "trde": "mfr",
    "id": "mfr",
    "ja": "mfr",
    "ko": "mfr",
    "th": "mfr",
    "vi": "mfr",
    "es": "mfr",
    "it": "hybrid_rf_byt5",
}

HYBRID_CONFIG = {
    # ES latest stable line from your experiments.
    "es": {
        "detector_path": "./detectors/es_change_detector_rf.joblib",
        "model_path": "./final_model/es_model",
        "threshold": 0.48,
        "mfr_min_conf": 0.4,#modified
    },
    # IT uses the same RF detector + MFR + ByT5 hybrid structure.
    # Keep it a little more conservative because IT had more case over-change risk.
    "it": {
        "detector_path": "./detectors/it_change_detector_rf.joblib",
        "model_path": "./final_model/it_model",
        "threshold": 0.75,
        "mfr_min_conf": 0.80,
    },
}

ALL_LANGS = sorted([
    "en", "da", "de", "es", "hr", "it", "nl", "sl", "sr", "tr",
    "iden", "trde", "id", "ja", "ko", "th", "vi",
])


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
         .replace("‘", "'")
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


def is_protected_token(token: str) -> bool:
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


def has_long_repetition(token: str) -> bool:
    return re.search(r"(.)\1{2,}", token.lower()) is not None


def has_laughter_pattern(token: str) -> bool:
    t = token.lower()
    return bool(re.search(r"(ja){2,}", t) or re.search(r"(jaja){1,}", t))


def token_shape(token: str) -> str:
    out = []
    for ch in token:
        base = strip_accents(ch)
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


def get_raw_stat_features(raw: str, raw_stats):
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


def get_key_stat_features(raw: str, key_stats):
    key = make_key(raw)
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


def detector_features(raw: str, left: str, right: str, raw_stats, key_stats):
    t = raw
    low = t.lower()
    key = make_key(t)
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    punct = sum((not ch.isalnum()) for ch in t)
    feats = {
        "bias": 1,
        "raw_lower=" + low: 1,
        "key=" + key: 1,
        "shape=" + token_shape(t): 1,
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
        "is_protected": int(is_protected_token(t)),
        "starts_at": int(t.startswith("@")),
        "starts_hash": int(t.startswith("#")),
        "starts_http": int(t.startswith("http://") or t.startswith("https://")),
        "is_digit_like": int(bool(re.fullmatch(r"\d+([.,:/-]\d+)*", t))),
        "has_long_repetition": int(has_long_repetition(t)),
        "has_laughter_pattern": int(has_laughter_pattern(t)),
        "has_accent": int(strip_accents(t) != t),
        "is_all_lower": int(t.islower()),
        "is_all_upper": int(t.isupper()),
        "is_title": int(t[:1].isupper() and t[1:].islower()),
        "has_qkx": int(any(ch in low for ch in ["q", "k", "x"])),
        "has_underscore": int("_" in t),
        "has_apostrophe": int("'" in normalize_apostrophe(t)),
    }
    feats.update(get_raw_stat_features(raw, raw_stats))
    feats.update(get_key_stat_features(raw, key_stats))
    return feats


def build_mfr_dictionary(train_rows) -> Tuple[Dict[str, str], Dict[str, float]]:
    counts = defaultdict(Counter)
    for row in train_rows:
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


def safe_generation(raw: str, pred: str) -> str:
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


def load_seq2seq_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def predict_western_row(lang: str, raw_words: List[str], fmt: str, tokenizer, model, device: str) -> List[str]:
    context = " ".join(raw_words)
    inputs_list = []
    for i, target_word in enumerate(raw_words):
        if fmt == "natural":
            inputs_list.append(f"lang: {lang} word: {target_word} context: {context}")
        else:
            words_copy = raw_words.copy()
            words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
            inputs_list.append(" ".join(words_copy))
    inputs = tokenizer(inputs_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=2)
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [safe_generation(raw, pred) for raw, pred in zip(raw_words, preds)]


def predict_hybrid_row(lang: str, raw_words: List[str], resources, device: str) -> List[str]:
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
        if is_protected_token(raw):
            pred_words[i] = raw
            continue

        mfr_pred = mfr.get(raw, raw)
        conf = mfr_conf.get(raw, 0.0)
        if mfr_pred != raw and conf >= mfr_min_conf:
            pred_words[i] = mfr_pred
            continue

        left = raw_words[i - 1] if i > 0 else "<BOS>"
        right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
        feats = detector_features(raw, left, right, raw_stats, key_stats)
        prob_change = detector_model.predict_proba([feats])[0][1]

        if prob_change < threshold or tokenizer is None or model is None:
            pred_words[i] = raw
            continue

        byt5_inputs.append(f"lang: {lang} word: {raw} context: {context}")
        byt5_positions.append(i)

    if byt5_inputs:
        inputs = tokenizer(byt5_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=2)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pos, raw, pred in zip(byt5_positions, [raw_words[p] for p in byt5_positions], preds):
            pred_words[pos] = safe_generation(raw, pred)

    return pred_words


def main():
    full_dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train_split = full_dataset["train"]
    eval_split = full_dataset["test"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("./submission_files", exist_ok=True)
    all_predictions_for_json = []

    print("\nFinal hybrid submission inference")
    print("device:", device)

    for lang in ALL_LANGS:
        lang_train = train_split.filter(lambda x: x["lang"] == lang)
        lang_eval = eval_split.filter(lambda x: x["lang"] == lang)
        if len(lang_eval) == 0:
            continue

        fmt = STRATEGY.get(lang, "mfr")
        print(f"\n[{lang.upper()}] mode={fmt}, eval rows={len(lang_eval)}")

        if fmt == "hybrid_rf_byt5":
            config = HYBRID_CONFIG[lang]
            mfr, mfr_conf = build_mfr_dictionary(lang_train)

            if not os.path.exists(config["detector_path"]):
                print(f"  detector missing -> fallback MFR: {config['detector_path']}")
                for row in tqdm(lang_eval, desc=f"[{lang.upper()}] MFR", leave=False):
                    raw_words = row["raw"]
                    pred_words = [mfr.get(w, w) for w in raw_words]
                    all_predictions_for_json.append({"raw": raw_words, "pred": pred_words, "lang": lang})
                continue

            detector = joblib.load(config["detector_path"])
            tokenizer, model = None, None
            if os.path.exists(config["model_path"]):
                tokenizer, model = load_seq2seq_model(config["model_path"], device)
            else:
                print(f"  ByT5 model missing -> detector can only copy/MFR: {config['model_path']}")

            resources = {
                "detector": detector,
                "threshold": config["threshold"],
                "mfr_min_conf": config["mfr_min_conf"],
                "mfr": mfr,
                "mfr_conf": mfr_conf,
                "tokenizer": tokenizer,
                "model": model,
            }

            for row in tqdm(lang_eval, desc=f"[{lang.upper()}] hybrid", leave=False):
                raw_words = row["raw"]
                pred_words = predict_hybrid_row(lang, raw_words, resources, device)
                all_predictions_for_json.append({"raw": raw_words, "pred": pred_words, "lang": lang})

            if model is not None:
                del model, tokenizer
                torch.cuda.empty_cache()
                gc.collect()

        elif fmt in {"natural", "sentinel"}:
            model_path = f"./final_model/{lang}_model"
            if not os.path.exists(model_path):
                print(f"  model missing -> fallback MFR: {model_path}")
                mfr, _ = build_mfr_dictionary(lang_train)
                for row in tqdm(lang_eval, desc=f"[{lang.upper()}] MFR", leave=False):
                    raw_words = row["raw"]
                    pred_words = [mfr.get(w, w) for w in raw_words]
                    all_predictions_for_json.append({"raw": raw_words, "pred": pred_words, "lang": lang})
                continue

            tokenizer, model = load_seq2seq_model(model_path, device)
            for row in tqdm(lang_eval, desc=f"[{lang.upper()}] ByT5", leave=False):
                raw_words = row["raw"]
                pred_words = predict_western_row(lang, raw_words, fmt, tokenizer, model, device)
                all_predictions_for_json.append({"raw": raw_words, "pred": pred_words, "lang": lang})
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        else:
            mfr, _ = build_mfr_dictionary(lang_train)
            for row in tqdm(lang_eval, desc=f"[{lang.upper()}] MFR", leave=False):
                raw_words = row["raw"]
                pred_words = [mfr.get(w, w) for w in raw_words]
                all_predictions_for_json.append({"raw": raw_words, "pred": pred_words, "lang": lang})

    json_path = "./submission_files/predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions_for_json, f, ensure_ascii=False)

    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")

    print("\nSaved:", json_path)
    print("Saved: submission.zip")


if __name__ == "__main__":
    main()
