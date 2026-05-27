import os
import re
import json
import gc
import zipfile
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


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

    # Newly developed languages
    "es": "marked_natural",
    "it": "marked_natural",
    "ko": "marked_natural",

    # Fallback languages
    "id": "mfr",
    "ja": "mfr",
    "th": "mfr",
    "vi": "mfr",
}


ALL_LANGS = sorted(
    [
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
)


KOREAN_RULE_CANDIDATES = {
    "낼": ["내일"],
    "넘": ["너무"],
    "걍": ["그냥"],
    "머": ["뭐"],
    "모해": ["뭐해"],
    "뭐해?": ["뭐해"],
    "마니": ["많이"],
    "조아": ["좋아"],
    "조앙": ["좋아"],
    "시러": ["싫어"],
    "싫엉": ["싫어"],
    "안뇽": ["안녕"],
    "방가": ["반가워"],
    "ㄱㅅ": ["감사", "고마워"],
    "ㅈㅅ": ["죄송", "미안"],
    "ㅇㅋ": ["오케이"],
    "ㄴㄴ": ["아니"],
    "ㅇㅇ": ["응"],
    "ㅁㅊ": ["미친"],
    "개웃겨": ["웃겨"],
}


def build_candidate_dictionary(train_data):
    """
    Store all observed norm candidates for each raw token.
    This extends MFR from one best answer to a candidate dictionary with confidence.
    """
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


def decompose_hangul_char(ch):
    base = ord(ch) - ord("가")

    if base < 0 or base > 11171:
        return ch

    chosung = [
        "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
        "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]
    jungsung = [
        "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
        "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
        "ㅡ", "ㅢ", "ㅣ",
    ]
    jongsung = [
        "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ",
        "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
        "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ]

    cho = base // 588
    jung = (base % 588) // 28
    jong = base % 28

    return chosung[cho] + jungsung[jung] + jongsung[jong]


def decompose_hangul_text(text):
    return "".join(decompose_hangul_char(ch) for ch in text)


def korean_distance(a, b):
    """
    Korean uses both syllable-level edit distance and jamo-level edit distance.
    """
    normal_dist = levenshtein_distance(a, b)

    decomposed_a = decompose_hangul_text(a)
    decomposed_b = decompose_hangul_text(b)
    jamo_dist = levenshtein_distance(decomposed_a, decomposed_b)

    return min(normal_dist, jamo_dist * 0.6)


def collapse_repeated_korean_emotes(token):
    candidates = []

    if re.fullmatch(r"[ㅋㅎㅠㅜ]+", token):
        if set(token) == {"ㅋ"} and len(token) >= 3:
            candidates.extend(["ㅋㅋ", "ㅋ"])

        if set(token) == {"ㅎ"} and len(token) >= 3:
            candidates.extend(["ㅎㅎ", "ㅎ"])

        if set(token) == {"ㅠ"} and len(token) >= 3:
            candidates.extend(["ㅠㅠ", "ㅠ"])

        if set(token) == {"ㅜ"} and len(token) >= 3:
            candidates.extend(["ㅜㅜ", "ㅜ"])

    return candidates


def get_korean_rule_candidates(raw_word):
    candidates = []

    if raw_word in KOREAN_RULE_CANDIDATES:
        candidates.extend(KOREAN_RULE_CANDIDATES[raw_word])

    candidates.extend(collapse_repeated_korean_emotes(raw_word))

    deduped = []
    for cand in candidates:
        if cand not in deduped:
            deduped.append(cand)

    return deduped


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


def choose_final_prediction_general(raw_word, model_candidates, cand_info):
    """
    General reranking for es/it and other model-based languages.
    """
    cleaned_candidates = []

    for pred in model_candidates:
        pred = clean_prediction(raw_word, pred)
        if pred not in cleaned_candidates:
            cleaned_candidates.append(pred)

    if len(cleaned_candidates) == 0:
        cleaned_candidates = [raw_word]

    model_top1 = cleaned_candidates[0]

    if raw_word not in cand_info:
        return model_top1

    info = cand_info[raw_word]
    train_candidates = [cand for cand, _ in info["candidates"]]

    total = info["total"]
    best = info["best"]
    best_ratio = info["best_ratio"]

    # Strong MFR if the training evidence is stable.
    if total >= 3 and best_ratio >= 0.90:
        return best

    # If a model candidate exactly appears in train candidates, use it.
    for pred in cleaned_candidates:
        if pred in train_candidates:
            return pred

    # If a model candidate is very close to a train candidate, correct to the train candidate.
    best_candidate = None
    best_dist = 999

    for pred in cleaned_candidates:
        for train_cand in train_candidates:
            dist = levenshtein_distance(pred, train_cand)
            if dist < best_dist:
                best_dist = dist
                best_candidate = train_cand

    if best_dist <= 1:
        return best_candidate

    # Medium-confidence MFR fallback.
    if total >= 2 and best_ratio >= 0.70:
        return best

    return model_top1


def choose_final_prediction_ko(raw_word, model_candidates, cand_info):
    """
    Korean-specific reranking using train candidates, rule candidates, and jamo distance.
    """
    cleaned_candidates = []

    for pred in model_candidates:
        pred = clean_prediction(raw_word, pred)
        if pred not in cleaned_candidates:
            cleaned_candidates.append(pred)

    if len(cleaned_candidates) == 0:
        cleaned_candidates = [raw_word]

    model_top1 = cleaned_candidates[0]

    train_candidates = []
    total = 0
    best = None
    best_ratio = 0.0

    if raw_word in cand_info:
        info = cand_info[raw_word]
        train_candidates = [cand for cand, _ in info["candidates"]]
        total = info["total"]
        best = info["best"]
        best_ratio = info["best_ratio"]

    rule_candidates = get_korean_rule_candidates(raw_word)

    all_known_candidates = []
    for cand in train_candidates + rule_candidates:
        if cand not in all_known_candidates:
            all_known_candidates.append(cand)

    # Strong MFR if the training evidence is stable.
    if best is not None and total >= 3 and best_ratio >= 0.90:
        return best

    # Exact match with train/rule candidates.
    for pred in cleaned_candidates:
        if pred in all_known_candidates:
            return pred

    # Jamo-aware correction.
    if len(all_known_candidates) > 0:
        best_candidate = None
        best_dist = 999

        for pred in cleaned_candidates:
            for cand in all_known_candidates:
                dist = korean_distance(pred, cand)
                if dist < best_dist:
                    best_dist = dist
                    best_candidate = cand

        if best_dist <= 1.2:
            return best_candidate

    # Rule candidate if training evidence is weak.
    if len(rule_candidates) == 1 and total <= 1:
        return rule_candidates[0]

    # Medium-confidence MFR fallback.
    if best is not None and total >= 2 and best_ratio >= 0.70:
        return best

    return model_top1


def choose_final_prediction_mfr_only(raw_word, cand_info):
    if raw_word not in cand_info:
        return raw_word

    return cand_info[raw_word]["best"]


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
    print("Loading dataset...", flush=True)
    full_dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    print("Dataset loaded.", flush=True)

    train_split = full_dataset["train"]
    eval_split = full_dataset["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    os.makedirs("./submission_files", exist_ok=True)
    all_predictions_for_json = []

    print("\n========== Inference with ES / IT / KO reranking ==========")

    for lang in ALL_LANGS:

        print(f"\n[{lang}] start", flush=True)
        print(f"[{lang}] filtering train/eval...", flush=True)
        lang_train = train_split.filter(lambda x: x["lang"] == lang)
        lang_eval = eval_split.filter(lambda x: x["lang"] == lang)

        if len(lang_eval) == 0:
            continue

        print(f"[{lang}] train rows={len(lang_train)}, eval rows={len(lang_eval)}", flush=True)
        print(f"[{lang}] building candidate dictionary...", flush=True)
        fmt = STRATEGY.get(lang, "mfr")
        cand_info = build_candidate_dictionary(lang_train)
        print(f"[{lang}] candidate dictionary done.", flush=True)
        use_deep_learning = fmt != "mfr"
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

        if not use_deep_learning:
            print(f"[{lang.upper()}] MFR only")

        for row in tqdm(lang_eval, desc=f"[{lang.upper()}] Predicting", leave=False):
            raw_words = row["raw"]

            if use_deep_learning:
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
                    if lang == "ko":
                        final_pred = choose_final_prediction_ko(
                            raw_word=raw_word,
                            model_candidates=model_candidates,
                            cand_info=cand_info,
                        )
                    else:
                        final_pred = choose_final_prediction_general(
                            raw_word=raw_word,
                            model_candidates=model_candidates,
                            cand_info=cand_info,
                        )

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
