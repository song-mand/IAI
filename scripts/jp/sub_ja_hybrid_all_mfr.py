#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict

import joblib
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ALL_LANGS = [
    "en",
    "da",
    "de",
    "es",
    "hr",
    "it",
    "nl",
    "sl",
    "sr",
    "tr",
    "iden",
    "trde",
    "id",
    "ja",
    "ko",
    "th",
    "vi",
]


# ============================================================
# Basic utilities
# ============================================================

def target_of(raw, norm):
    return norm if norm is not None else raw


def nfkc(s):
    return unicodedata.normalize("NFKC", s)


def is_hiragana(ch):
    return "\u3040" <= ch <= "\u309f"


def is_katakana(ch):
    return "\u30a0" <= ch <= "\u30ff"


def is_kanji(ch):
    return "\u4e00" <= ch <= "\u9fff"


def kata_to_hira(s):
    out = []

    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)

    return "".join(out)


def collapse_repeats(s, max_repeat=2):
    return re.sub(
        r"(.)\1{" + str(max_repeat) + r",}",
        lambda m: m.group(1) * max_repeat,
        s,
    )


def make_ja_key(s):
    s = nfkc(s)
    s = kata_to_hira(s)
    s = s.lower()
    s = collapse_repeats(s, max_repeat=2)
    return s


def is_protected_token(token):
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


def token_shape_ja(token):
    out = []

    for ch in token:
        if is_hiragana(ch):
            out.append("H")
        elif is_katakana(ch):
            out.append("K")
        elif is_kanji(ch):
            out.append("C")
        elif ch.isdigit():
            out.append("0")
        elif ch.isalpha():
            out.append("a")
        else:
            out.append(ch)

    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(out))


def has_long_repetition(token):
    return re.search(r"(.)\1{2,}", token) is not None


def has_kana(token):
    return any(is_hiragana(ch) or is_katakana(ch) for ch in token)


def has_kanji(token):
    return any(is_kanji(ch) for ch in token)


def has_hiragana(token):
    return any(is_hiragana(ch) for ch in token)


def has_katakana(token):
    return any(is_katakana(ch) for ch in token)


# ============================================================
# MFR
# ============================================================

def build_mfr_dictionary(train_data):
    counts = defaultdict(Counter)

    for row in train_data:
        raw_words = row["raw"]
        norm_words = row["norm"]

        for raw_word, norm_word in zip(raw_words, norm_words):
            target = target_of(raw_word, norm_word)
            counts[raw_word][target] += 1

    mfr = {}
    conf = {}

    for raw_word, counter in counts.items():
        total = sum(counter.values())

        best_target, best_count = max(
            counter.items(),
            key=lambda x: (x[1], x[0] == raw_word),
        )

        mfr[raw_word] = best_target
        conf[raw_word] = best_count / total if total else 0.0

    return mfr, conf


def predict_mfr(raw_words, mfr_dict):
    return [mfr_dict.get(w, w) for w in raw_words]


# ============================================================
# JA detector features
# ============================================================

def stat_features(raw, raw_stats, key_stats):
    feats = {}

    rs = raw_stats.get(raw)

    if rs is None:
        feats.update(
            {
                "raw_seen": 0,
                "raw_total": 0,
                "raw_change_prob": 0.0,
                "raw_copy_prob": 0.0,
                "raw_best_is_copy": 0,
                "raw_best_prob": 0.0,
            }
        )
    else:
        feats.update(
            {
                "raw_seen": 1,
                "raw_total": min(rs["total"], 10),
                "raw_change_prob": rs["change_prob"],
                "raw_copy_prob": rs["copy_prob"],
                "raw_best_is_copy": int(rs["best_norm"] == raw),
                "raw_best_prob": rs["best_prob"],
            }
        )

    key = make_ja_key(raw)
    ks = key_stats.get(key)

    if ks is None:
        feats.update(
            {
                "key_seen": 0,
                "key_total": 0,
                "key_best_prob": 0.0,
                "key_best_is_raw": 0,
            }
        )
    else:
        feats.update(
            {
                "key_seen": 1,
                "key_total": min(ks["total"], 10),
                "key_best_prob": ks["best_prob"],
                "key_best_is_raw": int(ks["best_norm"] == raw),
            }
        )

    return feats


def detector_features(raw, left, right, raw_stats, key_stats):
    n = nfkc(raw)
    key = make_ja_key(raw)

    chars = len(raw)
    hira = sum(is_hiragana(ch) for ch in raw)
    kata = sum(is_katakana(ch) for ch in raw)
    kanji = sum(is_kanji(ch) for ch in raw)
    digits = sum(ch.isdigit() for ch in raw)
    latin = sum(ch.isascii() and ch.isalpha() for ch in raw)
    punct = sum(not ch.isalnum() for ch in raw)

    feats = {
        "bias": 1,

        "raw=" + raw: 1,
        "nfkc=" + n: 1,
        "key=" + key: 1,
        "shape=" + token_shape_ja(raw): 1,

        "left_key=" + make_ja_key(left): 1,
        "right_key=" + make_ja_key(right): 1,

        "prefix1=" + key[:1]: 1,
        "prefix2=" + key[:2]: 1,
        "prefix3=" + key[:3]: 1,

        "suffix1=" + key[-1:]: 1,
        "suffix2=" + key[-2:]: 1,
        "suffix3=" + key[-3:]: 1,

        "len": min(chars, 30),
        "hira": min(hira, 30),
        "kata": min(kata, 30),
        "kanji": min(kanji, 30),
        "digits": min(digits, 30),
        "latin": min(latin, 30),
        "punct": min(punct, 30),

        "is_protected": int(is_protected_token(raw)),
        "has_hiragana": int(has_hiragana(raw)),
        "has_katakana": int(has_katakana(raw)),
        "has_kanji": int(has_kanji(raw)),
        "has_kana": int(has_kana(raw)),
        "has_long_repetition": int(has_long_repetition(raw)),
        "nfkc_changed": int(raw != n),
        "kata_to_hira_changed": int(kata_to_hira(n) != n),
        "all_katakana": int(kata > 0 and hira == 0 and kanji == 0),
        "all_hiragana": int(hira > 0 and kata == 0 and kanji == 0),
        "mixed_script": int(sum(x > 0 for x in [hira, kata, kanji, latin]) >= 2),
    }

    feats.update(stat_features(raw, raw_stats, key_stats))
    return feats


# ============================================================
# JA ByT5 prediction
# ============================================================

def build_ja_inputs(raw_words, indices, prompt_format):
    inputs = []

    for i in indices:
        raw = raw_words[i]

        if prompt_format == "natural":
            context = " ".join(raw_words)
            input_text = f"lang: ja word: {raw} context: {context}"

        elif prompt_format == "marked_natural":
            words_copy = list(raw_words)
            words_copy[i] = f"<extra_id_0> {raw} <extra_id_1>"
            marked_context = " ".join(words_copy)
            input_text = f"normalize lang: ja target: {raw} context: {marked_context}"

        else:
            words_copy = list(raw_words)
            words_copy[i] = f"<extra_id_0> {raw} <extra_id_1>"
            input_text = " ".join(words_copy)

        inputs.append(input_text)

    return inputs


def safe_byt5_output(raw, pred):
    if pred is None:
        return None

    p = pred.strip()

    if not p:
        return None

    bad_markers = [
        "lang:",
        "word:",
        "context:",
        "target:",
        "<extra_id",
        "extra_id",
    ]

    if any(m in p.lower() for m in bad_markers):
        return None

    if len(p) > max(20, len(raw) * 3):
        return None

    if " " in p:
        return None

    return p


def predict_ja_hybrid_row(
    raw_words,
    mfr_dict,
    mfr_conf,
    detector,
    raw_stats,
    key_stats,
    tokenizer,
    model,
    device,
    detector_threshold,
    mfr_min_conf,
    prompt_format,
    max_length,
    max_new_tokens,
    num_beams,
):
    pred_words = list(raw_words)
    byt5_indices = []

    for i, raw in enumerate(raw_words):
        if is_protected_token(raw):
            pred_words[i] = raw
            continue

        mfr_pred = mfr_dict.get(raw, raw)
        conf = mfr_conf.get(raw, 0.0)

        if mfr_pred != raw and conf >= mfr_min_conf:
            pred_words[i] = mfr_pred
            continue

        left = raw_words[i - 1] if i > 0 else "<BOS>"
        right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"

        feats = detector_features(raw, left, right, raw_stats, key_stats)
        prob_change = detector.predict_proba([feats])[0][1]

        if prob_change >= detector_threshold:
            byt5_indices.append(i)
        else:
            pred_words[i] = raw

    if not byt5_indices:
        return pred_words

    inputs = build_ja_inputs(raw_words, byt5_indices, prompt_format)

    batch = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for idx, decoded_pred in zip(byt5_indices, decoded):
        raw = raw_words[idx]
        pred = safe_byt5_output(raw, decoded_pred)

        if pred is None:
            pred_words[idx] = raw
        elif pred != raw:
            pred_words[idx] = pred
        else:
            pred_words[idx] = raw

    return pred_words


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="weerayut/multilexnorm2026-dev-pub")

    parser.add_argument("--ja_model_dir", type=str, default="./final_model/ja_model")
    parser.add_argument("--ja_detector", type=str, default="./detectors/ja_change_detector_rf.joblib")

    parser.add_argument("--output_json", type=str, default="./submission_files/predictions.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip")

    parser.add_argument("--detector_threshold", type=float, default=-1.0)
    parser.add_argument("--mfr_min_conf", type=float, default=0.65)

    parser.add_argument(
        "--prompt_format",
        choices=["sentinel", "natural", "marked_natural"],
        default="sentinel",
    )

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name)

    train_split = dataset["train"]
    test_split = dataset["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    print("Building MFR dictionaries for all languages...")

    mfr_by_lang = {}
    conf_by_lang = {}

    for lang in ALL_LANGS:
        lang_train = train_split.filter(lambda x, lang=lang: x["lang"] == lang)

        mfr, conf = build_mfr_dictionary(lang_train)
        mfr_by_lang[lang] = mfr
        conf_by_lang[lang] = conf

        print(f"[{lang.upper()}] MFR entries: {len(mfr)}")

    use_ja_hybrid = (
        os.path.isdir(args.ja_model_dir)
        and os.path.isfile(args.ja_detector)
    )

    if use_ja_hybrid:
        print("[JA] Loading hybrid resources...")

        artifact = joblib.load(args.ja_detector)

        ja_detector = artifact["model"]
        ja_raw_stats = artifact["raw_stats"]
        ja_key_stats = artifact["key_stats"]

        if args.detector_threshold >= 0:
            ja_threshold = args.detector_threshold
        else:
            ja_threshold = artifact.get("threshold", 0.55)

        tokenizer = AutoTokenizer.from_pretrained(args.ja_model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.ja_model_dir).to(device)
        model.eval()

        print(f"[JA] model: {args.ja_model_dir}")
        print(f"[JA] detector: {args.ja_detector}")
        print(f"[JA] detector_threshold: {ja_threshold}")

    else:
        print("[JA] hybrid resources not found. JA will use MFR only.")

        ja_detector = None
        ja_raw_stats = None
        ja_key_stats = None
        ja_threshold = None
        tokenizer = None
        model = None

    predictions = []

    print("\nPredicting test split...")

    for row in tqdm(test_split, desc="Submission prediction"):
        lang = row["lang"]
        raw_words = list(row["raw"])

        if lang == "ja" and use_ja_hybrid:
            pred_words = predict_ja_hybrid_row(
                raw_words=raw_words,
                mfr_dict=mfr_by_lang.get("ja", {}),
                mfr_conf=conf_by_lang.get("ja", {}),
                detector=ja_detector,
                raw_stats=ja_raw_stats,
                key_stats=ja_key_stats,
                tokenizer=tokenizer,
                model=model,
                device=device,
                detector_threshold=ja_threshold,
                mfr_min_conf=args.mfr_min_conf,
                prompt_format=args.prompt_format,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        else:
            pred_words = predict_mfr(raw_words, mfr_by_lang.get(lang, {}))

        predictions.append(
            {
                "raw": raw_words,
                "pred": pred_words,
                "lang": lang,
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.output_json, arcname="predictions.json")

    if model is not None:
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\nSaved:")
    print(f"  {args.output_json}")
    print(f"  {args.output_zip}")


if __name__ == "__main__":
    main()