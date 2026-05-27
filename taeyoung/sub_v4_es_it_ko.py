import os
import re
import json
import gc
import zipfile
import math
import unicodedata
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

"""
sub_v4_es_it_ko.py

V4 direction:

1. Korean:
   - No hard-coded slang dictionary.
   - No ByT5 generation.
   - Train a lightweight copy/change classifier from the Korean train split at runtime.
   - If classifier says KEEP, keep raw.
   - If classifier says CHANGE, choose from train-derived candidates and jamo-distance candidates.
   - If confidence is weak, keep raw.

2. Spanish / Italian:
   - Conservative v3 logic is retained.
   - ByT5 is only a cautious candidate generator.
   - Strong copy-protection prevents overcorrection.

3. Other languages:
   - Same fallback style: model if available, otherwise MFR.
"""

STRATEGY = {
    "da": "natural", "en": "natural", "sl": "natural", "sr": "natural", "hr": "natural", "iden": "natural",
    "de": "sentinel", "nl": "sentinel", "tr": "sentinel", "trde": "sentinel",
    "es": "marked_natural", "it": "marked_natural",
    "ko": "ko_classifier_ranker",
    "id": "mfr", "ja": "mfr", "th": "mfr", "vi": "mfr",
}

ALL_LANGS = sorted(["en", "da", "de", "es", "hr", "it", "nl", "sl", "sr", "tr", "iden", "trde", "id", "ja", "ko", "th", "vi"])


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_for_compare(text):
    return strip_accents(str(text).lower())


def build_candidate_dictionary(train_data):
    cand_counts = {}
    for row in train_data:
        for raw_word, norm_word in zip(row["raw"], row["norm"]):
            target = norm_word if norm_word is not None else raw_word
            cand_counts.setdefault(raw_word, {})
            cand_counts[raw_word][target] = cand_counts[raw_word].get(target, 0) + 1

    cand_info = {}
    for raw_word, targets in cand_counts.items():
        total = sum(targets.values())
        sorted_targets = sorted(targets.items(), key=lambda x: (x[1], x[0] == raw_word), reverse=True)
        best_target, best_count = sorted_targets[0]
        raw_count = targets.get(raw_word, 0)
        cand_info[raw_word] = {
            "total": total,
            "candidates": sorted_targets,
            "best": best_target,
            "best_count": best_count,
            "best_ratio": best_count / total,
            "raw_count": raw_count,
            "raw_ratio": raw_count / total,
        }
    return cand_info


def build_norm_vocabulary(train_data):
    vocab_counts = {}
    for row in train_data:
        for raw_word, norm_word in zip(row["raw"], row["norm"]):
            target = norm_word if norm_word is not None else raw_word
            if target:
                vocab_counts[target] = vocab_counts.get(target, 0) + 1
    return vocab_counts


def build_norm_vocab_index(vocab_counts, key_func=None):
    index = defaultdict(list)
    for token, count in vocab_counts.items():
        if not token:
            continue
        if key_func is None:
            first = normalize_for_compare(token[:1])
        else:
            first = key_func(token)
        index[(first, len(token))].append((token, count))
    return index


def levenshtein_distance(a, b):
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def roman_distance(a, b):
    d1 = levenshtein_distance(str(a).lower(), str(b).lower())
    d2 = levenshtein_distance(normalize_for_compare(a), normalize_for_compare(b))
    return min(d1, d2)


# ---------- Korean jamo utilities ----------

CHOSUNG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNGSUNG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONGSUNG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def decompose_hangul_char(ch):
    base = ord(ch) - ord("가")
    if base < 0 or base > 11171:
        return ch
    cho = base // 588
    jung = (base % 588) // 28
    jong = base % 28
    return CHOSUNG[cho] + JUNGSUNG[jung] + JONGSUNG[jong]


def decompose_hangul_text(text):
    return "".join(decompose_hangul_char(ch) for ch in str(text))


def first_jamo_key(token):
    if not token:
        return ""
    decomposed = decompose_hangul_text(token[:1])
    return decomposed[:1] if decomposed else ""


def korean_distance(a, b):
    normal_dist = levenshtein_distance(a, b)
    jamo_dist = levenshtein_distance(decompose_hangul_text(a), decompose_hangul_text(b))
    return min(normal_dist, jamo_dist * 0.6)


def is_special_or_keep_token(raw_word):
    if raw_word is None or raw_word == "":
        return True
    if str(raw_word).startswith("@") or str(raw_word).startswith("#") or str(raw_word).startswith("http"):
        return True
    if re.fullmatch(r"[0-9]+([:.,/-][0-9]+)*", str(raw_word)):
        return True
    return False


def clean_prediction(raw_word, pred_word):
    if pred_word is None:
        return raw_word
    pred_word = str(pred_word).strip()
    if pred_word == "" or pred_word in ["<unk>", "<pad>", "</s>"]:
        return raw_word
    if len(pred_word) > len(str(raw_word)) * 3 + 5:
        return raw_word
    if is_special_or_keep_token(raw_word):
        return raw_word
    return pred_word


# ---------- Korean classifier + ranker ----------

def ko_instance_text(raw_words, idx):
    raw = raw_words[idx]
    prev_tok = raw_words[idx - 1] if idx > 0 else "<BOS>"
    next_tok = raw_words[idx + 1] if idx + 1 < len(raw_words) else "<EOS>"
    context = " ".join(raw_words[max(0, idx - 3): idx] + ["<T>", raw, "</T>"] + raw_words[idx + 1: idx + 4])
    # Include decomposed jamo to help classifier learn Korean noisy patterns without hard-coded mappings.
    return f"raw={raw} jamo={decompose_hangul_text(raw)} prev={prev_tok} next={next_tok} ctx={context}"


def train_ko_change_classifier(ko_train):
    texts, labels = [], []
    for row in ko_train:
        raw_words = row["raw"]
        norm_words = row["norm"]
        for i, (raw, norm) in enumerate(zip(raw_words, norm_words)):
            target = norm if norm is not None else raw
            texts.append(ko_instance_text(raw_words, i))
            labels.append(1 if target != raw else 0)

    if not SKLEARN_AVAILABLE:
        print("[KO] sklearn is not available. Falling back to candidate-only conservative mode.")
        return None

    if len(set(labels)) < 2:
        print("[KO] Only one class in labels. Falling back to candidate-only conservative mode.")
        return None

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=1, max_features=200000)),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0, solver="liblinear")),
    ])
    clf.fit(texts, labels)
    print(f"[KO] Trained copy/change classifier on {len(texts)} token instances. Change ratio={sum(labels)/len(labels):.4f}")
    return clf


def ko_change_probability(clf, raw_words, idx):
    if clf is None:
        return 0.0
    text = ko_instance_text(raw_words, idx)
    try:
        return float(clf.predict_proba([text])[0][1])
    except Exception:
        pred = int(clf.predict([text])[0])
        return 1.0 if pred == 1 else 0.0


def get_ko_norm_vocab_candidates(raw_word, vocab_index, max_candidates=8):
    key = first_jamo_key(raw_word)
    q_len = len(raw_word)
    collected = {}
    for length in range(max(1, q_len - 2), q_len + 3):
        for cand, count in vocab_index.get((key, length), []):
            if cand == raw_word:
                continue
            dist = korean_distance(raw_word, cand)
            # Conservative retrieval threshold.
            if len(raw_word) <= 3:
                ok = dist <= 1.2
            else:
                ok = dist <= 1.8
            if ok:
                score = (dist, -math.log1p(count), len(cand))
                prev = collected.get(cand)
                if prev is None or score < prev:
                    collected[cand] = score
    ranked = sorted(collected.items(), key=lambda x: x[1])
    return [cand for cand, _ in ranked[:max_candidates]]


def choose_final_prediction_ko_classifier_ranker(raw_words, idx, cand_info, ko_clf, ko_vocab_index):
    raw_word = raw_words[idx]
    if is_special_or_keep_token(raw_word):
        return raw_word

    p_change = ko_change_probability(ko_clf, raw_words, idx)
    info = cand_info.get(raw_word)

    # If train strongly says copy, copy regardless of classifier.
    if info is not None:
        if info["raw_ratio"] >= 0.70:
            return raw_word
        # If train gives a stable non-raw correction, trust it even with moderate classifier confidence.
        if info["best"] != raw_word and info["total"] >= 2 and info["best_ratio"] >= 0.80 and p_change >= 0.35:
            return info["best"]

    # If classifier does not think this token should change, keep raw.
    if p_change < 0.60:
        return raw_word

    candidates = []
    # Raw-specific candidates from train.
    if info is not None:
        for cand, count in info["candidates"]:
            if cand != raw_word:
                candidates.append((cand, "raw_candidate", count))

    # Jamo-distance norm vocabulary candidates, only when classifier indicates change.
    for cand in get_ko_norm_vocab_candidates(raw_word, ko_vocab_index, max_candidates=8):
        candidates.append((cand, "vocab_candidate", 1))

    if not candidates:
        return raw_word

    # Deduplicate with best source/count.
    merged = {}
    for cand, source, count in candidates:
        if cand not in merged or count > merged[cand][1]:
            merged[cand] = (source, count)

    best_cand = None
    best_score = None
    for cand, (source, count) in merged.items():
        dist = korean_distance(raw_word, cand)
        # Candidate ranker score: lower is better.
        # Strongly prefer raw-specific candidates over global vocab retrieval.
        source_penalty = 0.0 if source == "raw_candidate" else 0.8
        freq_bonus = -0.15 * math.log1p(count)
        change_bonus = -0.5 * p_change
        length_penalty = 0.15 * abs(len(cand) - len(raw_word))
        score = dist + source_penalty + freq_bonus + length_penalty + change_bonus
        if best_score is None or score < best_score:
            best_score = score
            best_cand = cand

    # Final safety gates.
    if best_cand is None:
        return raw_word

    best_dist = korean_distance(raw_word, best_cand)
    if info is None:
        # For unseen raw, require higher confidence and closer candidate.
        if p_change < 0.75:
            return raw_word
        if len(raw_word) <= 3 and best_dist > 1.2:
            return raw_word
        if len(raw_word) > 3 and best_dist > 1.8:
            return raw_word

    return best_cand


# ---------- ES / IT conservative v3 logic ----------

def build_norm_vocab_candidates_es_it(raw_word, model_candidates, vocab_index, max_candidates=5):
    query_terms = [raw_word] + list(model_candidates)
    collected = {}
    for query in query_terms:
        if not query:
            continue
        query = str(query).strip()
        q_first = normalize_for_compare(query[:1])
        q_len = len(query)
        for length in range(max(1, q_len - 1), q_len + 2):
            for cand, count in vocab_index.get((q_first, length), []):
                dist = roman_distance(query, cand)
                if dist <= 1:
                    score = (dist, -count)
                    prev = collected.get(cand)
                    if prev is None or score < prev:
                        collected[cand] = score
    ranked = sorted(collected.items(), key=lambda x: x[1])
    return [cand for cand, _ in ranked[:max_candidates]]


def choose_final_prediction_es_it(raw_word, model_candidates, cand_info, vocab_index):
    if is_special_or_keep_token(raw_word):
        return raw_word

    cleaned_candidates = []
    for pred in model_candidates:
        pred = clean_prediction(raw_word, pred)
        if pred not in cleaned_candidates:
            cleaned_candidates.append(pred)
    if not cleaned_candidates:
        cleaned_candidates = [raw_word]

    raw_info = cand_info.get(raw_word)
    raw_cmp = normalize_for_compare(raw_word)

    if raw_info is not None:
        total = raw_info["total"]
        best = raw_info["best"]
        best_ratio = raw_info["best_ratio"]
        raw_ratio = raw_info["raw_ratio"]
        train_candidates = [cand for cand, _ in raw_info["candidates"]]

        if raw_ratio >= 0.70:
            return raw_word
        if best != raw_word and total >= 3 and best_ratio >= 0.85:
            return best
        for pred in cleaned_candidates:
            if pred in train_candidates:
                return pred
        best_candidate, best_dist = None, 999
        for pred in cleaned_candidates:
            for train_cand in train_candidates:
                if train_cand == raw_word:
                    continue
                dist = roman_distance(pred, train_cand)
                if dist < best_dist:
                    best_dist = dist
                    best_candidate = train_cand
        if best_candidate is not None and best_dist <= 1:
            return best_candidate
        if best != raw_word and total >= 2 and best_ratio >= 0.75 and raw_ratio < 0.50:
            return best
        return raw_word

    # Unseen raw: very conservative.
    non_raw_candidates = [c for c in cleaned_candidates if normalize_for_compare(c) != raw_cmp]
    if len(non_raw_candidates) == 0:
        return raw_word

    vocab_candidates = build_norm_vocab_candidates_es_it(raw_word, cleaned_candidates, vocab_index, max_candidates=5)
    for cand in vocab_candidates:
        if normalize_for_compare(cand) == raw_cmp:
            continue
        dist_to_raw = roman_distance(raw_word, cand)
        dist_to_model = min(roman_distance(cand, pred) for pred in cleaned_candidates)
        if len(raw_word) <= 4:
            if dist_to_raw <= 1 and dist_to_model <= 1:
                return cand
        else:
            if dist_to_raw <= 2 and dist_to_model <= 1:
                return cand
    return raw_word


# ---------- Shared model inference ----------

def get_model_path_for_lang(lang):
    model_path = f"./final_model/{lang}_model"
    return model_path if os.path.exists(model_path) else None


def make_inputs(fmt, lang, raw_words):
    inputs_list = []
    plain_context = " ".join(raw_words)
    for i, target_word in enumerate(raw_words):
        if fmt == "marked_natural":
            words_copy = raw_words.copy()
            words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
            marked_context = " ".join(words_copy)
            input_text = f"normalize lang: {lang} target: {target_word} context: {marked_context}"
        elif fmt == "natural":
            input_text = f"lang: {lang} word: {target_word} context: {plain_context}"
        elif fmt == "sentinel":
            words_copy = raw_words.copy()
            words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
            input_text = " ".join(words_copy)
        else:
            input_text = target_word
        inputs_list.append(input_text)
    return inputs_list


def predict_candidates_with_model(model, tokenizer, device, fmt, lang, raw_words, num_beams=5, num_return_sequences=3):
    inputs_list = make_inputs(fmt, lang, raw_words)
    inputs = tokenizer(inputs_list, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded = [x.strip() for x in decoded]
    return [decoded[i:i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]


def choose_final_prediction_mfr_only(raw_word, cand_info):
    if raw_word not in cand_info:
        return raw_word
    return cand_info[raw_word]["best"]


def main():
    full_dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train_split = full_dataset["train"]
    eval_split = full_dataset["test"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("./submission_files", exist_ok=True)
    all_predictions_for_json = []

    print("\n========== Inference v4: KO classifier/ranker + conservative ES/IT ==========")

    # Train Korean classifier once.
    ko_train = train_split.filter(lambda x: x["lang"] == "ko")
    ko_classifier = train_ko_change_classifier(ko_train)

    for lang in ALL_LANGS:
        lang_train = train_split.filter(lambda x: x["lang"] == lang)
        lang_eval = eval_split.filter(lambda x: x["lang"] == lang)
        if len(lang_eval) == 0:
            continue

        fmt = STRATEGY.get(lang, "mfr")
        cand_info = build_candidate_dictionary(lang_train)

        vocab_index = None
        if lang in ["es", "it"]:
            vocab_index = build_norm_vocab_index(build_norm_vocabulary(lang_train))
        elif lang == "ko":
            vocab_index = build_norm_vocab_index(build_norm_vocabulary(lang_train), key_func=first_jamo_key)

        use_deep_learning = fmt not in ["mfr", "ko_classifier_ranker"]
        model_path = get_model_path_for_lang(lang) if use_deep_learning else None
        tokenizer = None
        model = None

        if use_deep_learning and model_path is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                model.eval()
                print(f"[{lang.upper()}] Model inference: {model_path}")
            except Exception as e:
                print(f"[{lang.upper()}] Model load failed. Fallback to MFR. {e}")
                use_deep_learning = False
        else:
            use_deep_learning = False

        if lang == "ko":
            print(f"[{lang.upper()}] Copy/change classifier + candidate ranker")
        elif not use_deep_learning:
            print(f"[{lang.upper()}] MFR only")

        for row in tqdm(lang_eval, desc=f"[{lang.upper()}] Predicting", leave=False):
            raw_words = row["raw"]

            if lang == "ko":
                pred_words = [
                    choose_final_prediction_ko_classifier_ranker(
                        raw_words=raw_words,
                        idx=i,
                        cand_info=cand_info,
                        ko_clf=ko_classifier,
                        ko_vocab_index=vocab_index,
                    )
                    for i in range(len(raw_words))
                ]

            elif use_deep_learning:
                grouped_model_candidates = predict_candidates_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    fmt=fmt,
                    lang=lang,
                    raw_words=raw_words,
                    num_beams=5,
                    num_return_sequences=3,
                )
                pred_words = []
                for raw_word, model_candidates in zip(raw_words, grouped_model_candidates):
                    if lang in ["es", "it"]:
                        final_pred = choose_final_prediction_es_it(raw_word, model_candidates, cand_info, vocab_index)
                    else:
                        final_pred = choose_final_prediction_mfr_only(raw_word, cand_info)
                        if final_pred == raw_word and model_candidates:
                            final_pred = clean_prediction(raw_word, model_candidates[0])
                    pred_words.append(final_pred)

            else:
                pred_words = [choose_final_prediction_mfr_only(raw_word, cand_info) for raw_word in raw_words]

            all_predictions_for_json.append({"raw": raw_words, "pred": pred_words, "lang": lang})

        if use_deep_learning:
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    json_path = "./submission_files/predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions_for_json, f, ensure_ascii=False)

    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="predictions.json")

    print("\n========== Done ==========")
    print("Saved: submission.zip")


if __name__ == "__main__":
    main()
