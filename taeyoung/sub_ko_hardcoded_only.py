import json
import zipfile
from datasets import load_dataset
from tqdm import tqdm
from ko_hardcode_rules import build_ko_candidate_info, choose_ko_hardcoded

"""
sub_ko_hardcoded_only.py

This script creates a Korean-only predictions file for quick testing.
It is useful when you want to inspect only Korean behavior.
It outputs:
- ./submission_files/predictions_ko_only.json
- submission_ko_only.zip

For the full multilingual submission, copy the Korean branch from this file
into your main sub script or use sub_v5_es_it_ko_hardcoded.py.
"""

AGGRESSIVE = False


def main():
    dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train_split = dataset["train"]
    test_split = dataset["test"]

    ko_train = train_split.filter(lambda x: x["lang"] == "ko")
    ko_test = test_split.filter(lambda x: x["lang"] == "ko")
    ko_info = build_ko_candidate_info(ko_train)

    preds = []
    for row in tqdm(ko_test, desc="[KO] hardcoded prediction"):
        raw_words = row["raw"]
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
        preds.append({"raw": raw_words, "pred": pred_words, "lang": "ko"})

    import os
    os.makedirs("./submission_files", exist_ok=True)
    json_path = "./submission_files/predictions_ko_only.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False)

    with zipfile.ZipFile("submission_ko_only.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(json_path, arcname="predictions.json")

    print("Saved: submission_ko_only.zip")


if __name__ == "__main__":
    main()
