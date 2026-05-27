import os
import json
import zipfile
from collections import Counter, defaultdict

from datasets import load_dataset
from tqdm import tqdm

from ko_hardcode_rules import build_ko_candidate_info, choose_ko_hardcoded


"""
sub_all_mfr_ko_hardcoded.py

Full multilingual submission script.

Strategy:
- ko: Korean hardcoded rules + train candidate info
- all other languages: MFR only

Outputs:
- ./submission_files/predictions.json
- ./submission.zip
"""


AGGRESSIVE = False

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


def build_mfr_dictionary(train_data):
    """
    Build raw -> most frequent norm dictionary.

    If there is a tie, prefer copying raw itself.
    This helps reduce unnecessary over-change.
    """
    counts = defaultdict(Counter)

    for row in train_data:
        for raw_word, norm_word in zip(row["raw"], row["norm"]):
            target = norm_word if norm_word is not None else raw_word
            counts[raw_word][target] += 1

    mfr = {}

    for raw_word, counter in counts.items():
        best_target, _ = max(
            counter.items(),
            key=lambda x: (x[1], x[0] == raw_word),
        )
        mfr[raw_word] = best_target

    return mfr


def predict_mfr(raw_words, mfr_dict):
    return [mfr_dict.get(w, w) for w in raw_words]


def main():
    print("Loading dataset...")
    dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")

    train_split = dataset["train"]
    test_split = dataset["test"]

    os.makedirs("./submission_files", exist_ok=True)

    print("Building language resources...")

    mfr_by_lang = {}
    ko_info = None

    for lang in ALL_LANGS:
        lang_train = train_split.filter(lambda x, lang=lang: x["lang"] == lang)

        if lang == "ko":
            print("[KO] building hardcoded candidate info")
            ko_info = build_ko_candidate_info(lang_train)
        else:
            print(f"[{lang.upper()}] building MFR dictionary")
            mfr_by_lang[lang] = build_mfr_dictionary(lang_train)

    all_predictions = []

    print("\nPredicting full test set...")

    for row in tqdm(test_split, desc="Predicting"):
        lang = row["lang"]
        raw_words = row["raw"]

        if lang == "ko":
            pred_words = [
                choose_ko_hardcoded(
                    raw_word=w,
                    ko_info=ko_info,
                    aggressive=AGGRESSIVE,
                    trust_train=True,
                    raw_keep_threshold=0.70,
                    change_threshold=0.80,
                )
                for w in raw_words
            ]
        else:
            mfr_dict = mfr_by_lang.get(lang, {})
            pred_words = predict_mfr(raw_words, mfr_dict)

        all_predictions.append(
            {
                "raw": raw_words,
                "pred": pred_words,
                "lang": lang,
            }
        )

    json_path = "./submission_files/predictions.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)

    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="predictions.json")

    print("\nSaved:")
    print("  ./submission_files/predictions.json")
    print("  ./submission.zip")


if __name__ == "__main__":
    main()