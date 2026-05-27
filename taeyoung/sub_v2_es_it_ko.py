import os
import re
import json
import gc
import zipfile
import unicodedata
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

"""
sub_v2_es_it_ko.py

Main changes from the previous version:

1. Korean:
   - Do NOT use ByT5 generation.
   - Use conservative MFR + safe Korean rules only.
   - Unknown Korean tokens are kept unchanged.

2. Spanish / Italian:
   - Use ByT5 models if available.
   - Generate multiple model candidates.
   - Rerank with:
     a) raw-specific train candidates
     b) MFR confidence
     c) language-specific norm vocabulary candidates
     d) edit-distance / accent-insensitive distance
     e) overcorrection protection

3. Other languages:
   - Keep the original strategy style.
   - If a trained model exists, it can be used.
   - Otherwise fallback to MFR.
"""

STRATEGY = {
    "da": "natural",
    "en": "natural",
    "sl": "natural",
    "sr": "natural",
    "hr": "natural",
    "iden": "natural",

    "de": "sentinel",
    "nl": "sentinel",
    "tr": "sentinel",
    "trde": "sentinel",

    # Developed with model + candidate reranking
    "es": "marked_natural",
    "it": "marked_natural",

    # Korean v2: conservative rule-based MFR, no ByT5 generation
    "ko": "ko_rule_mfr",

    # Fallback languages
    "id": "mfr",
    "ja": "mfr",
    "th": "mfr",
    "vi": "mfr",
}

ALL_LANGS = sorted(
    [
        "en", "da", "de", "es", "hr", "it", "nl", "sl", "sr",
        "tr", "iden", "trde", "id", "ja", "ko", "th", "vi",
    ]
)

# Conservative Korean rules only.
# Avoid ambiguous rules such as ㄱㅅ -> 감사/고마워 unless it is observed in train.
KOREAN_SAFE_RULES = {
    "낼": "내일",
    "넘": "너무",
    "걍": "그냥",
    "머": "뭐",
    "모해": "뭐해",
    "뭐해?": "뭐해",
    "마니": "많이",
    "조아": "좋아",
    "조앙": "좋아",
    "시러": "싫어",
    "싫엉": "싫어",
    "안뇽": "안녕",
    "방가": "반가워",
    "ㅇㅋ": "오케이",
    "ㄴㄴ": "아니",
    "ㅇㅇ": "응",
}


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def build_candidate_dictionary(train_data):
    cand_counts = {}

    for row in train_data:
        for raw_word, norm_word in zip(row["raw"], row["norm"]):
            target = norm_word if norm_word is not None else raw_word

            if raw_word not in cand_counts:
                cand_counts[raw_word] = {}

            cand_counts[raw_word][target] = cand_counts[raw_word].get(target, 0) + 1

    cand_info = {}

    for raw_word, targets in cand_counts.items():
        total = sum(targets.values())
        sorted_targets = sorted(
            targets.items(),
            key=lambda x: (x[1], x[0] == raw_word),
            reverse=True,
        )

        best_target, best_count = sorted_targets[0]

        cand_info[raw_word] = {
            "total": total,
            "candidates": sorted_targets,
            "best": best_target,
            "best_count": best_count,
            "best_ratio": best_count / total,
        }

    return cand_info


def build_norm_vocabulary(train_data):
    vocab_counts = {}

    for row in train_data:
        for raw_word, norm_word in zip(row["raw"], row["norm"]):
            target = norm_word if norm_word is not None else raw_word
            if target is None:
                continue
            vocab_counts[target] = vocab_counts.get(target, 0) + 1

    return vocab_counts


def build_norm_vocab_index(vocab_counts):
    index = defaultdict(list)

    for token, count in vocab_counts.items():
        if token is None or token == "":
            continue

        key_first = strip_accents(token[:1].lower())
        key_len = len(token)

        index[(key_first, key_len)].append((token, count))

    return index


def levenshtein_distance(a, b):
    if a is None:
        a = ""
    if b is None:
        b = ""

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
    d1 = levenshtein_distance(a.lower(), b.lower())
    d2 = levenshtein_distance(strip_accents(a.lower()), strip_accents(b.lower()))
    return min(d1, d2)


def get_norm_vocab_candidates(raw_word, model_candidates, vocab_index, max_candidates=8):
    query_terms = [raw_word] + list(model_candidates)
    collected = {}

    for query in query_terms:
        if query is None:
            continue

        query = query.strip()
        if query == "":
            continue

        q_first = strip_accents(query[:1].lower())
        q_len = len(query)

        for length in range(max(1, q_len - 2), q_len + 3):
            bucket = vocab_index.get((q_first, length), [])

            for cand, count in bucket:
                dist = roman_distance(query, cand)

                if len(query) <= 4:
                    ok = dist <= 1
                else:
                    ok = dist <= 2

                if ok:
                    prev = collected.get(cand)
                    score = (dist, -count)
                    if prev is None or score < prev:
                        collected[cand] = score

    ranked = sorted(collected.items(), key=lambda x: x[1])
    return [cand for cand, _ in ranked[:max_candidates]]


def clean_prediction(raw_word, pred_word):
    if pred_word is None:
        return raw_word

    pred_word = pred_word.strip()

    if pred_word == "":
        return raw_word

    if pred_word in ["<unk>", "<pad>", "</s>"]:
        return raw_word

    if len(pred_word) > len(raw_word) * 3 + 5:
        return raw_word

    if raw_word.startswith("@") or raw_word.startswith("#") or raw_word.startswith("http"):
        return raw_word

    return pred_word


def is_special_or_keep_token(raw_word):
    if raw_word is None or raw_word == "":
        return True

    if raw_word.startswith("@") or raw_word.startswith("#") or raw_word.startswith("http"):
        return True

    if re.fullmatch(r"[0-9]+([:.,/-][0-9]+)*", raw_word):
        return True

    return False


def overcorrection_guard(raw_word, candidate, support_level):
    if candidate == raw_word:
        return candidate

    if is_special_or_keep_token(raw_word):
        return raw_word

    dist = roman_distance(raw_word, candidate)

    if support_level == "strong":
        return candidate

    if support_level == "medium":
        if len(raw_word) <= 4:
            return candidate if dist <= 2 else raw_word
        return candidate if dist <= max(2, int(len(raw_word) * 0.5)) else raw_word

    # model-only candidate should be conservative
    if len(raw_word) <= 4:
        return candidate if dist <= 1 else raw_word

    return candidate if dist <= max(2, int(len(raw_word) * 0.4)) else raw_word


def choose_final_prediction_es_it(raw_word, model_candidates, cand_info, vocab_index):
    if is_special_or_keep_token(raw_word):
        return raw_word

    cleaned_candidates = []
    for pred in model_candidates:
        pred = clean_prediction(raw_word, pred)
        if pred not in cleaned_candidates:
            cleaned_candidates.append(pred)

    if len(cleaned_candidates) == 0:
        cleaned_candidates = [raw_word]

    model_top1 = cleaned_candidates[0]
    raw_info = cand_info.get(raw_word)

    if raw_info is not None:
        train_candidates = [cand for cand, _ in raw_info["candidates"]]
        total = raw_info["total"]
        best = raw_info["best"]
        best_ratio = raw_info["best_ratio"]

        # strong raw-specific MFR
        if total >= 3 and best_ratio >= 0.90:
            return best

        # exact model candidate in raw-specific candidates
        for pred in cleaned_candidates:
            if pred in train_candidates:
                return pred

        # close to raw-specific train candidates
        best_candidate = None
        best_dist = 999

        for pred in cleaned_candidates:
            for train_cand in train_candidates:
                dist = roman_distance(pred, train_cand)
                if dist < best_dist:
                    best_dist = dist
                    best_candidate = train_cand

        if best_candidate is not None and best_dist <= 1:
            return best_candidate

    # global norm vocabulary candidate search
    vocab_candidates = get_norm_vocab_candidates(
        raw_word=raw_word,
        model_candidates=cleaned_candidates,
        vocab_index=vocab_index,
        max_candidates=8,
    )

    if len(vocab_candidates) > 0:
        best_vocab_candidate = None
        best_score = None

        for cand in vocab_candidates:
            dist_to_model = min(roman_distance(cand, pred) for pred in cleaned_candidates)
            dist_to_raw = roman_distance(cand, raw_word)
            score = (dist_to_model, dist_to_raw, len(cand))

            if best_score is None or score < best_score:
                best_score = score
                best_vocab_candidate = cand

        guarded = overcorrection_guard(raw_word, best_vocab_candidate, support_level="medium")
        if guarded != raw_word:
            return guarded

    # medium raw-specific MFR fallback
    if raw_info is not None:
        total = raw_info["total"]
        best = raw_info["best"]
        best_ratio = raw_info["best_ratio"]

        if total >= 2 and best_ratio >= 0.70:
            return best

    # model top-1 with overcorrection guard
    return overcorrection_guard(raw_word, model_top1, support_level="weak")


def choose_final_prediction_mfr_only(raw_word, cand_info):
    if raw_word not in cand_info:
        return raw_word
    return cand_info[raw_word]["best"]


def choose_final_prediction_ko_rule_mfr(raw_word, cand_info):
    if raw_word is None or raw_word == "":
        return raw_word

    if raw_word.startswith("@") or raw_word.startswith("#") or raw_word.startswith("http"):
        return raw_word

    # If observed in train, trust raw-specific MFR.
    if raw_word in cand_info:
        return cand_info[raw_word]["best"]

    # Safe direct rules only for unseen tokens.
    if raw_word in KOREAN_SAFE_RULES:
        return KOREAN_SAFE_RULES[raw_word]

    # Repeated Korean laughing/crying expressions are risky.
    # Keep unchanged unless observed in train.
    if re.fullmatch(r"[ㅋㅎㅠㅜ]+", raw_word):
        return raw_word

    return raw_word


def get_model_path_for_lang(lang):
    model_path = f"./final_model/{lang}_model"
    if os.path.exists(model_path):
        return model_path
    return None


def make_inputs(fmt, lang, raw_words):
    inputs_list = []
    plain_context = " ".join(raw_words)

    for i, target_word in enumerate(raw_words):
        if fmt == "marked_natural":
            words_copy = raw_words.copy()
            words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
            marked_context = " ".join(words_copy)

            input_text = (
                f"normalize lang: {lang} "
                f"target: {target_word} "
                f"context: {marked_context}"
            )

        elif fmt == "natural":
            input_text = (
                f"lang: {lang} "
                f"word: {target_word} "
                f"context: {plain_context}"
            )

        elif fmt == "sentinel":
            words_copy = raw_words.copy()
            words_copy[i] = f"<extra_id_0> {target_word} <extra_id_1>"
            input_text = " ".join(words_copy)

        else:
            input_text = target_word

        inputs_list.append(input_text)

    return inputs_list


def predict_candidates_with_model(
    model,
    tokenizer,
    device,
    fmt,
    lang,
    raw_words,
    num_beams=5,
    num_return_sequences=3,
):
    inputs_list = make_inputs(fmt, lang, raw_words)

    inputs = tokenizer(
        inputs_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

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

    grouped = []
    for i in range(0, len(decoded), num_return_sequences):
        grouped.append(decoded[i : i + num_return_sequences])

    return grouped


def main():
    full_dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train_split = full_dataset["train"]
    eval_split = full_dataset["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("./submission_files", exist_ok=True)
    all_predictions_for_json = []

    print("\n========== Inference v2: ES/IT norm-vocab rerank + KO conservative MFR ==========")

    for lang in ALL_LANGS:
        lang_train = train_split.filter(lambda x: x["lang"] == lang)
        lang_eval = eval_split.filter(lambda x: x["lang"] == lang)

        if len(lang_eval) == 0:
            continue

        fmt = STRATEGY.get(lang, "mfr")
        cand_info = build_candidate_dictionary(lang_train)

        vocab_index = None
        if lang in ["es", "it"]:
            vocab_counts = build_norm_vocabulary(lang_train)
            vocab_index = build_norm_vocab_index(vocab_counts)

        use_deep_learning = fmt not in ["mfr", "ko_rule_mfr"]
        model_path = None

        if use_deep_learning:
            model_path = get_model_path_for_lang(lang)

        tokenizer = None
        model = None

        if use_deep_learning and model_path is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                model.eval()
                print(f"[{lang.upper()}] Model inference: {model_path}")
            except Exception as e:
                print(f"[{lang.upper()}] Model load failed. Fallback to MFR.")
                print(e)
                use_deep_learning = False
        else:
            use_deep_learning = False

        if fmt == "ko_rule_mfr":
            print(f"[{lang.upper()}] Conservative Korean MFR + safe rules")
        elif not use_deep_learning:
            print(f"[{lang.upper()}] MFR only")

        for row in tqdm(lang_eval, desc=f"[{lang.upper()}] Predicting", leave=False):
            raw_words = row["raw"]

            if fmt == "ko_rule_mfr":
                pred_words = [
                    choose_final_prediction_ko_rule_mfr(raw_word, cand_info)
                    for raw_word in raw_words
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
                        final_pred = choose_final_prediction_es_it(
                            raw_word=raw_word,
                            model_candidates=model_candidates,
                            cand_info=cand_info,
                            vocab_index=vocab_index,
                        )
                    else:
                        # For other model-based languages, use model top-1 with simple cleanup.
                        final_pred = clean_prediction(raw_word, model_candidates[0])

                    pred_words.append(final_pred)

            else:
                pred_words = [
                    choose_final_prediction_mfr_only(raw_word, cand_info)
                    for raw_word in raw_words
                ]

            all_predictions_for_json.append(
                {
                    "raw": raw_words,
                    "pred": pred_words,
                    "lang": lang,
                }
            )

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
