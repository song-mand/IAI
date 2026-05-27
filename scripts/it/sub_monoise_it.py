import argparse
import json
import os
import re
import shlex
import subprocess
import unicodedata
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "weerayut/multilexnorm2026-dev-pub"
IT_LANG = "it"

TARGET_LANGS = sorted([
    "en", "da", "de", "es", "hr", "it", "nl", "sl", "sr", "tr",
    "iden", "trde", "id", "ja", "ko", "th", "vi"
])


URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
MENTION_RE = re.compile(r"^@\w+")
HASHTAG_RE = re.compile(r"^#\S+")
PUNCT_ONLY_RE = re.compile(r"^[^\wÀ-ÖØ-öø-ÿ]+$")
REPEAT_RE = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ])\1{2,}")


SMS_ABBREV_SET = {
    "nn", "nnt", "cmq", "cqm", "ke", "ki", "x", "xke", "xké", "xkè",
    "xche", "xchè", "xché", "perke", "perké", "perkè", "grz", "qnd",
    "cn", "dv", "qst", "qsto", "qsta", "qsti", "qste", "tt", "sn",
    "sx", "dx", "info", "nov",
}


def safe_norm(raw: str, norm: Any) -> str:
    return raw if norm is None else str(norm)


def get_lang_rows(split, lang: str) -> List[Dict[str, Any]]:
    return [row for row in split if row["lang"] == lang]


def strip_accents_and_marks(s: str) -> str:
    s = str(s)
    s = s.replace("’", "'").replace("`", "'").replace("´", "'")
    s = s.replace("'", "")

    decomposed = unicodedata.normalize("NFD", s)
    without_accents = "".join(
        ch for ch in decomposed
        if unicodedata.category(ch) != "Mn"
    )

    return without_accents.lower()


def is_protected_token(tok: str) -> bool:
    return (
        bool(URL_RE.match(tok))
        or bool(MENTION_RE.match(tok))
        or bool(HASHTAG_RE.match(tok))
        or bool(PUNCT_ONLY_RE.match(tok))
    )


def build_mfr_counts(rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    counts = defaultdict(Counter)

    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            raw = str(raw)
            norm = safe_norm(raw, norm)
            counts[raw][norm] += 1

    return counts


def choose_mfr(raw: str, counter: Counter) -> str:
    if not counter:
        return raw

    max_count = max(counter.values())
    bests = [k for k, v in counter.items() if v == max_count]

    if raw in bests:
        return raw

    return sorted(bests)[0]


def build_mfr_dict(mfr_counts: Dict[str, Counter]) -> Dict[str, str]:
    return {
        raw: choose_mfr(raw, counter)
        for raw, counter in mfr_counts.items()
    }


def get_mapping_stats(raw: str, cand: str, mfr_counts: Dict[str, Counter]) -> Tuple[int, int, float]:
    counter = mfr_counts.get(raw, Counter())
    total = sum(counter.values())
    count = counter.get(cand, 0)
    conf = count / total if total else 0.0
    return count, total, conf


def is_case_only(raw: str, cand: str) -> bool:
    return raw.lower() == cand.lower() and raw != cand


def is_diacritic_mapping(raw: str, cand: str) -> bool:
    if raw == cand:
        return False

    if raw.lower() == cand.lower():
        return False

    return strip_accents_and_marks(raw) == strip_accents_and_marks(cand)


def is_repeat_mapping(raw: str, cand: str) -> bool:
    if REPEAT_RE.search(raw) is None:
        return False

    reduced_one = REPEAT_RE.sub(r"\1", raw)
    reduced_two = REPEAT_RE.sub(r"\1\1", raw)

    return cand in {reduced_one, reduced_two}


def classify_mapping(raw: str, cand: str) -> str:
    if raw == cand:
        return "copy"

    if is_protected_token(raw):
        return "protected"

    if " " in cand:
        return "multiword"

    if is_case_only(raw, cand):
        if raw.isupper() and cand.islower():
            return "allcaps_lower"
        if raw.isupper() and cand[:1].isupper() and cand[1:].islower():
            return "allcaps_title"
        if raw[:1].isupper() and cand.islower():
            return "title_lower"
        if raw.islower() and cand[:1].isupper():
            return "lower_title"
        return "case_only"

    if is_diacritic_mapping(raw, cand):
        return "diacritic"

    if is_repeat_mapping(raw, cand):
        return "repeat"

    raw_low = raw.lower()

    if raw_low in SMS_ABBREV_SET:
        return "sms_abbrev"

    if len(cand) >= len(raw) + 4:
        return "expansion"

    if len(raw) <= 3 and len(cand) > len(raw):
        return "short_expansion"

    return "spelling_or_other"


def write_monoise_input(rows: List[Dict[str, Any]], path: str) -> None:
    """
    MoNoise에 넣을 입력.
    기본 가정: tokenized sentence one line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(map(str, row["raw"])) + "\n")


def read_monoise_output(path: str, rows: List[Dict[str, Any]]) -> List[List[str]]:
    """
    기본 가정: MoNoise output도 tokenized sentence one line.
    길이가 안 맞으면 해당 문장은 raw copy로 fallback.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if len(lines) != len(rows):
        raise RuntimeError(
            f"MoNoise output line count mismatch: output={len(lines)}, rows={len(rows)}"
        )

    preds = []

    for line, row in zip(lines, rows):
        raw_words = [str(x) for x in row["raw"]]
        pred_words = line.split()

        if len(pred_words) != len(raw_words):
            pred_words = raw_words.copy()

        preds.append(pred_words)

    return preds


def run_monoise_command(input_path: str, output_path: str) -> None:
    """
    환경변수 MONOISE_CMD를 사용한다.

    예:
    export MONOISE_CMD='python /path/to/monoise/normalize.py --model /path/to/it_model --input {input} --output {output}'

    {input}, {output} placeholder는 반드시 들어가야 한다.
    """
    cmd_template = os.environ.get("MONOISE_CMD", "").strip()

    if not cmd_template:
        raise RuntimeError(
            "MONOISE_CMD is not set. Example:\n"
            "export MONOISE_CMD='python /path/to/monoise/normalize.py "
            "--model /path/to/it_model --input {input} --output {output}'"
        )

    if "{input}" not in cmd_template or "{output}" not in cmd_template:
        raise RuntimeError("MONOISE_CMD must contain {input} and {output} placeholders.")

    cmd = cmd_template.format(
        input=shlex.quote(input_path),
        output=shlex.quote(output_path),
    )

    print("[MoNoise] command:")
    print(cmd)

    subprocess.run(cmd, shell=True, check=True)


def load_or_create_monoise_predictions(
    rows: List[Dict[str, Any]],
    work_dir: str,
    cache_json: str,
    run_monoise: bool,
) -> List[List[str]]:
    os.makedirs(work_dir, exist_ok=True)

    if os.path.exists(cache_json) and not run_monoise:
        print(f"[MoNoise] load cache: {cache_json}")
        with open(cache_json, "r", encoding="utf-8") as f:
            return json.load(f)

    input_path = os.path.join(work_dir, "it_monoise_input.txt")
    output_path = os.path.join(work_dir, "it_monoise_output.txt")

    write_monoise_input(rows, input_path)

    if run_monoise:
        run_monoise_command(input_path, output_path)
    else:
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"MoNoise output not found: {output_path}\n"
                "Either run with --run-monoise or put output file there."
            )

    preds = read_monoise_output(output_path, rows)

    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"[MoNoise] saved cache: {cache_json}")

    return preds


MONOISE_POLICIES = {
    "m0_mfr": {
        "description": "plain MFR baseline",
    },

    "m1_monoise_only": {
        "description": "IT uses MoNoise directly; others MFR",
    },

    "m2_mfr_preserve_unseen_safe": {
        "description": "Preserve MFR changes; apply MoNoise only when MFR copies raw and MoNoise change is safe",
        "allowed_categories": {
            "diacritic",
            "repeat",
            "sms_abbrev",
        },
    },

    "m3_mfr_preserve_unseen_diacritic": {
        "description": "Preserve MFR changes; apply MoNoise only for unseen diacritic changes",
        "allowed_categories": {
            "diacritic",
        },
    },

    "m4_mfr_or_monoise_agree": {
        "description": "Use MFR if it changes; otherwise use MoNoise only for safe categories",
        "allowed_categories": {
            "diacritic",
            "repeat",
            "sms_abbrev",
            "spelling_or_other",
        },
    },

    "m5_mfr_monoise_agreement_only": {
        "description": "Apply only when MFR and MoNoise agree on a changed form; otherwise MFR",
    },
}


def accept_monoise_candidate(
    raw: str,
    monoise_pred: str,
    mode: str,
) -> Tuple[bool, str, str]:
    if monoise_pred == raw:
        return False, "copy", "monoise_copy"

    category = classify_mapping(raw, monoise_pred)

    if category == "protected":
        return False, category, "protected"

    policy = MONOISE_POLICIES[mode]
    allowed_categories = policy.get("allowed_categories")

    if allowed_categories is None:
        return True, category, "allowed_no_category_filter"

    if category in allowed_categories:
        return True, category, "allowed_category"

    return False, category, f"category_not_allowed:{category}"


def predict_it_with_monoise(
    raw_words: List[str],
    monoise_words: List[str],
    mfr_dict: Dict[str, str],
    mode: str,
    debug_collector: List[Dict[str, Any]] = None,
) -> List[str]:
    pred = []

    for i, raw in enumerate(raw_words):
        raw = str(raw)
        mfr = mfr_dict.get(raw, raw)
        mono = monoise_words[i] if i < len(monoise_words) else raw

        if mode == "m0_mfr":
            final = mfr
            source = "mfr"

        elif mode == "m1_monoise_only":
            final = mono
            source = "monoise"

        elif mode == "m5_mfr_monoise_agreement_only":
            # MFR과 MoNoise가 같은 변경을 제안하면 적용.
            # 그 외에는 기존 MFR 유지.
            if mfr != raw and mono == mfr:
                final = mfr
                source = "mfr_monoise_agree"
            else:
                final = mfr
                source = "mfr_default"

        else:
            # 핵심 모드:
            # MFR이 이미 바꾸는 token은 MFR 보존.
            # MFR이 copy하는 token에만 MoNoise 보조 후보를 본다.
            if mfr != raw:
                final = mfr
                source = "mfr_preserved"
            else:
                accepted, category, reason = accept_monoise_candidate(
                    raw=raw,
                    monoise_pred=mono,
                    mode=mode,
                )

                if accepted:
                    final = mono
                    source = f"monoise_accepted:{category}"
                else:
                    final = raw
                    source = f"copy:{reason}"

        pred.append(final)

        if debug_collector is not None:
            if raw != final or raw != mono or raw != mfr:
                debug_collector.append({
                    "raw": raw,
                    "mfr": mfr,
                    "monoise": mono,
                    "final": final,
                    "source": source,
                    "category_mfr": classify_mapping(raw, mfr),
                    "category_monoise": classify_mapping(raw, mono),
                    "prev": raw_words[i - 1] if i > 0 else "<BOS>",
                    "next": raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>",
                    "sentence": raw_words,
                })

    return pred


def predict_mfr_sentence(raw_words: List[str], mfr_dict: Dict[str, str]) -> List[str]:
    return [mfr_dict.get(str(w), str(w)) for w in raw_words]


def analyze_predictions(
    lang: str,
    rows: List[Dict[str, Any]],
    pred_rows: List[List[str]],
    debug_items: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    total = 0
    changed = 0

    change_counter = Counter()
    source_counter = Counter()
    category_counter = Counter()
    samples = []

    for row, pred in zip(rows, pred_rows):
        raw_words = [str(x) for x in row["raw"]]
        changes = []

        for raw, p in zip(raw_words, pred):
            total += 1

            if raw != p:
                changed += 1
                change_counter[(raw, p)] += 1
                changes.append((raw, p))

        if changes and len(samples) < 30:
            samples.append({
                "raw": raw_words,
                "pred": pred,
                "changes": changes,
            })

    if debug_items:
        for item in debug_items:
            source_counter[item["source"]] += 1
            if item["final"] != item["raw"]:
                category_counter[classify_mapping(item["raw"], item["final"])] += 1

    return {
        "lang": lang,
        "total_tokens": total,
        "changed_tokens": changed,
        "changed_rate": changed / total if total else 0.0,
        "top_changes": [
            {"raw": r, "pred": p, "count": c}
            for (r, p), c in change_counter.most_common(100)
        ],
        "source_counts": dict(source_counter),
        "changed_category_counts": dict(category_counter),
        "samples": samples,
    }


def print_analysis(analysis: Dict[str, Any], detail: bool = False):
    print(
        f"[{analysis['lang'].upper():5s}] "
        f"tokens={analysis['total_tokens']:6d} "
        f"changed={analysis['changed_tokens']:5d} "
        f"rate={analysis['changed_rate'] * 100:6.2f}%"
    )

    if not detail:
        return

    print("\n[IT] Top changes")
    for item in analysis["top_changes"][:80]:
        print(f"  {item['raw']} -> {item['pred']} | count={item['count']}")

    print("\n[IT] Source counts")
    print(json.dumps(analysis["source_counts"], ensure_ascii=False, indent=2))

    print("\n[IT] Changed category counts")
    print(json.dumps(analysis["changed_category_counts"], ensure_ascii=False, indent=2))

    print("\n[IT] Changed samples")
    for i, sample in enumerate(analysis["samples"][:15], start=1):
        print(f"\n  sample {i}")
        print("  raw : " + " ".join(map(str, sample["raw"])))
        print("  pred: " + " ".join(map(str, sample["pred"])))
        print(f"  changes: {sample['changes']}")


def print_monoise_debug(debug_items: List[Dict[str, Any]], max_items: int = 100):
    print("\n[IT] MoNoise decision debug")

    for item in debug_items[:max_items]:
        print(
            f"  raw={item['raw']} | mfr={item['mfr']} | monoise={item['monoise']} | "
            f"final={item['final']} | source={item['source']} | "
            f"cat_mfr={item['category_mfr']} | cat_monoise={item['category_monoise']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--mode", default="m2_mfr_preserve_unseen_safe", choices=sorted(MONOISE_POLICIES.keys()))
    parser.add_argument("--out-dir", default="scripts/it/monoise_outputs")
    parser.add_argument("--zip-name", default=None)
    parser.add_argument("--run-monoise", action="store_true")
    parser.add_argument("--debug-it", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.zip_name is None:
        args.zip_name = f"submission_{args.mode}.zip"

    print("=" * 60)
    print("1. Load dataset")
    print("=" * 60)

    ds = load_dataset(args.dataset)
    train_split = ds["train"]
    test_split = ds["test"]

    print(f"mode: {args.mode}")
    print(f"description: {MONOISE_POLICIES[args.mode]['description']}")

    print("=" * 60)
    print("2. Build MFR dictionaries")
    print("=" * 60)

    mfr_dict_by_lang = {}

    for lang in TARGET_LANGS:
        lang_train = get_lang_rows(train_split, lang)
        mfr_counts = build_mfr_counts(lang_train)
        mfr_dict = build_mfr_dict(mfr_counts)
        mfr_dict_by_lang[lang] = mfr_dict

        print(f"[{lang.upper():5s}] MFR entries: {len(mfr_dict)}")

    print("=" * 60)
    print("3. Prepare MoNoise predictions for IT")
    print("=" * 60)

    it_rows = get_lang_rows(test_split, IT_LANG)
    monoise_cache = os.path.join(args.out_dir, "it_monoise_test_predictions.json")

    monoise_preds = load_or_create_monoise_predictions(
        rows=it_rows,
        work_dir=args.out_dir,
        cache_json=monoise_cache,
        run_monoise=args.run_monoise,
    )

    print("=" * 60)
    print("4. Predict")
    print("=" * 60)

    predictions = []
    diagnostics = {
        "method": "IT=MFR+MoNoise-candidate-gate, others=MFR",
        "mode": args.mode,
        "policy": MONOISE_POLICIES[args.mode],
        "languages": {},
    }

    test_langs = sorted(set(row["lang"] for row in test_split))

    for lang in test_langs:
        rows = get_lang_rows(test_split, lang)
        pred_rows = []
        it_debug_items = []

        print("\n" + "-" * 60)

        if lang == IT_LANG:
            print(f"[{lang.upper()}] Predict with MFR + MoNoise gate: {args.mode}")
        else:
            print(f"[{lang.upper()}] Predict with plain MFR")

        print("-" * 60)

        for idx, row in enumerate(tqdm(rows, desc=f"[{lang.upper()}] predict")):
            raw_words = [str(x) for x in row["raw"]]

            if lang == IT_LANG:
                pred = predict_it_with_monoise(
                    raw_words=raw_words,
                    monoise_words=monoise_preds[idx],
                    mfr_dict=mfr_dict_by_lang[lang],
                    mode=args.mode,
                    debug_collector=it_debug_items,
                )
            else:
                pred = predict_mfr_sentence(
                    raw_words=raw_words,
                    mfr_dict=mfr_dict_by_lang.get(lang, {}),
                )

            if len(pred) != len(raw_words):
                print(f"[WARN] Length mismatch in {lang}. Fallback to raw copy.")
                pred = raw_words.copy()

            pred_rows.append(pred)

            predictions.append({
                "raw": raw_words,
                "pred": pred,
                "lang": lang,
            })

        analysis = analyze_predictions(
            lang=lang,
            rows=rows,
            pred_rows=pred_rows,
            debug_items=it_debug_items if lang == IT_LANG else None,
        )

        diagnostics["languages"][lang] = analysis

        print_analysis(analysis, detail=(lang == IT_LANG))

        if lang == IT_LANG and args.debug_it:
            diagnostics["it_monoise_decision_debug"] = it_debug_items
            print_monoise_debug(it_debug_items, max_items=120)

    print("=" * 60)
    print("5. Save outputs")
    print("=" * 60)

    pred_json = os.path.join(args.out_dir, f"predictions_{args.mode}.json")
    diag_json = os.path.join(args.out_dir, f"diagnostics_{args.mode}.json")
    zip_path = os.path.join(args.out_dir, args.zip_name)

    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_json, arcname="predictions.json")

    print(f"predictions: {pred_json}")
    print(f"diagnostics: {diag_json}")
    print(f"zip:         {zip_path}")

    print("=" * 60)
    print("6. Zip check")
    print("=" * 60)

    with zipfile.ZipFile(zip_path, "r") as zipf:
        for info in zipf.infolist():
            print(f"{info.filename} | {info.file_size} bytes")

    print("Done.")


if __name__ == "__main__":
    main()