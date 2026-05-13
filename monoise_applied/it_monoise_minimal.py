import argparse
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


LANG = "it"


IT_ABBREV = {
    "nn": ["non"],
    "nnt": ["niente"],
    "cmq": ["comunque"],
    "Cmq": ["Comunque"],
    "qnd": ["quando"],
    "qst": ["questo", "questa", "questi", "queste"],
    "x": ["per"],
    "X": ["Per"],
    "xke": ["perché"],
    "xké": ["perché"],
    "xchè": ["perché"],
    "ke": ["che"],
    "k": ["che"],
    "sn": ["sono"],
    "dv": ["dove"],
    "tt": ["tutto", "tutti", "tutte"],
    "anke": ["anche"],
    "bn": ["bene"],
    "info": ["informazioni"],
    "e'": ["è"],
    "E'": ["È"],
    "perche'": ["perché"],
    "perchè": ["perché"],
    "piu'": ["più"],
    "li'": ["lì"],
    "po": ["po'"],
}


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_apostrophe(s: str) -> str:
    return s.replace("’", "'").replace("`", "'").replace("´", "'")


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


def contains_alpha(s: str) -> bool:
    return any(ch.isalpha() for ch in s)


def char_order_preserved(raw: str, cand: str) -> bool:
    raw_l = raw.lower()
    cand_l = cand.lower()
    j = 0
    for ch in cand_l:
        if j < len(raw_l) and raw_l[j] == ch:
            j += 1
    return j == len(raw_l)


def edit_distance(a: str, b: str, max_len: int = 64) -> int:
    a = a[:max_len]
    b = b[:max_len]
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


class ITCandidateGenerator:
    """Minimal MoNoise-style candidate generator for Italian.

    Candidate modules:
      1. original token
      2. train lookup raw -> observed norm
      3. normalized-key lookup
      4. case candidate from train norm vocabulary
      5. abbreviation/accent/apostrophe rule candidates
      6. repeated-character collapse candidate
    """

    def __init__(self, top_k_per_source: int = 8):
        self.top_k_per_source = top_k_per_source
        self.exact = defaultdict(Counter)
        self.key_index = defaultdict(Counter)
        self.case_index = defaultdict(Counter)
        self.norm_vocab = Counter()
        self.unigram = Counter()
        self.bigram_left = Counter()
        self.bigram_right = Counter()

    def fit(self, rows):
        for row in rows:
            if row["lang"] != LANG:
                continue
            raw_words = row["raw"]
            norm_words = [
                n if n is not None else r
                for r, n in zip(row["raw"], row["norm"])
            ]
            for i, (raw, target) in enumerate(zip(raw_words, norm_words)):
                self.exact[raw][target] += 1
                self.key_index[make_key(raw)][target] += 1
                self.case_index[target.lower()][target] += 1
                self.norm_vocab[target] += 1
                left = raw_words[i - 1] if i > 0 else "<BOS>"
                right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
                self.unigram[target] += 1
                self.bigram_left[(left.lower(), target)] += 1
                self.bigram_right[(target, right.lower())] += 1
        return self

    def _add(self, table, cand, source):
        if cand is None or cand == "":
            return
        table[cand].add(source)

    def _add_counter_top(self, table, counter, source, top_k=None):
        if top_k is None:
            top_k = self.top_k_per_source
        for cand, _ in counter.most_common(top_k):
            self._add(table, cand, source)

    def generate(self, raw: str):
        candidates = defaultdict(set)

        if is_protected_token(raw):
            self._add(candidates, raw, "original")
            return candidates

        self._add(candidates, raw, "original")

        if raw in self.exact:
            self._add_counter_top(candidates, self.exact[raw], "lookup")
            best, _ = self.exact[raw].most_common(1)[0]
            self._add(candidates, best, "mfr")

        key = make_key(raw)
        if key in self.key_index:
            self._add_counter_top(candidates, self.key_index[key], "key")

        low = raw.lower()
        if low in self.case_index:
            self._add_counter_top(candidates, self.case_index[low], "case")

        if raw in IT_ABBREV:
            for cand in IT_ABBREV[raw]:
                self._add(candidates, cand, "rule")

        raw_lower = raw.lower()
        if raw_lower in IT_ABBREV:
            for cand in IT_ABBREV[raw_lower]:
                self._add(candidates, cand, "rule_lower")

        collapsed1 = collapse_repeats(raw, max_repeat=1)
        if collapsed1 != raw and collapsed1 in self.norm_vocab:
            self._add(candidates, collapsed1, "repeat")

        collapsed2 = collapse_repeats(raw, max_repeat=2)
        if collapsed2 != raw and collapsed2 in self.norm_vocab:
            self._add(candidates, collapsed2, "repeat")

        return candidates


def get_lookup_stats(gen: ITCandidateGenerator, raw: str, cand: str):
    raw_counter = gen.exact.get(raw, Counter())
    raw_total = sum(raw_counter.values())
    pair_count = raw_counter.get(cand, 0)
    copy_count = raw_counter.get(raw, 0)
    pair_conf = pair_count / raw_total if raw_total else 0.0
    copy_conf = copy_count / raw_total if raw_total else 0.0

    key_counter = gen.key_index.get(make_key(raw), Counter())
    key_total = sum(key_counter.values())
    key_count = key_counter.get(cand, 0)
    key_conf = key_count / key_total if key_total else 0.0

    case_counter = gen.case_index.get(raw.lower(), Counter())
    case_total = sum(case_counter.values())
    case_count = case_counter.get(cand, 0)
    case_conf = case_count / case_total if case_total else 0.0

    return {
        "raw_total": raw_total,
        "pair_count": pair_count,
        "pair_conf": pair_conf,
        "copy_count": copy_count,
        "copy_conf": copy_conf,
        "key_count": key_count,
        "key_conf": key_conf,
        "case_count": case_count,
        "case_conf": case_conf,
    }


def feature_dict(raw: str, cand: str, left: str, right: str, sources, gen: ITCandidateGenerator):
    stats = get_lookup_stats(gen, raw, cand)
    raw_key = make_key(raw)
    cand_key = make_key(cand)
    dist = edit_distance(raw.lower(), cand.lower())
    max_len = max(len(raw), len(cand), 1)

    feats = {
        "bias": 1,
        "is_original": int(cand == raw),
        "from_original": int("original" in sources),
        "from_lookup": int("lookup" in sources),
        "from_mfr": int("mfr" in sources),
        "from_key": int("key" in sources),
        "from_case": int("case" in sources),
        "from_rule": int("rule" in sources or "rule_lower" in sources),
        "from_repeat": int("repeat" in sources),
        "num_sources": len(sources),
        "raw_seen": int(stats["raw_total"] > 0),
        "raw_total_log": math.log1p(stats["raw_total"]),
        "pair_count_log": math.log1p(stats["pair_count"]),
        "pair_conf": stats["pair_conf"],
        "copy_count_log": math.log1p(stats["copy_count"]),
        "copy_conf": stats["copy_conf"],
        "key_count_log": math.log1p(stats["key_count"]),
        "key_conf": stats["key_conf"],
        "case_count_log": math.log1p(stats["case_count"]),
        "case_conf": stats["case_conf"],
        "cand_freq_log": math.log1p(gen.norm_vocab.get(cand, 0)),
        "unigram_log": math.log1p(gen.unigram.get(cand, 0)),
        "left_bigram_log": math.log1p(gen.bigram_left.get((left.lower(), cand), 0)),
        "right_bigram_log": math.log1p(gen.bigram_right.get((cand, right.lower()), 0)),
        "same_string": int(raw == cand),
        "same_lower": int(raw.lower() == cand.lower()),
        "same_key": int(raw_key == cand_key),
        "same_accentless": int(strip_accents(raw).lower() == strip_accents(cand).lower()),
        "edit_distance": min(dist, 10),
        "norm_edit_distance": dist / max_len,
        "len_raw": min(len(raw), 40),
        "len_cand": min(len(cand), 40),
        "len_diff": min(abs(len(raw) - len(cand)), 40),
        "chars_order_preserved": int(char_order_preserved(raw, cand)),
        "contains_alpha_raw": int(contains_alpha(raw)),
        "contains_alpha_cand": int(contains_alpha(cand)),
        "raw_has_accent": int(strip_accents(raw) != raw),
        "cand_has_accent": int(strip_accents(cand) != cand),
        "accent_added_or_fixed": int(strip_accents(raw).lower() == strip_accents(cand).lower() and raw != cand),
        "raw_has_apostrophe": int("'" in normalize_apostrophe(raw)),
        "cand_has_apostrophe": int("'" in normalize_apostrophe(cand)),
        "apostrophe_changed": int(normalize_apostrophe(raw) != normalize_apostrophe(cand)),
        "raw_all_upper": int(raw.isupper()),
        "raw_all_lower": int(raw.islower()),
        "raw_title": int(raw.istitle()),
        "cand_all_upper": int(cand.isupper()),
        "cand_all_lower": int(cand.islower()),
        "cand_title": int(cand.istitle()),
        "left_lower=" + left.lower(): 1,
        "right_lower=" + right.lower(): 1,
        "raw_lower=" + raw.lower(): 1,
        "cand_lower=" + cand.lower(): 1,
    }
    return feats


def build_ranker_examples(rows, gen: ITCandidateGenerator, force_gold=True):
    X = []
    y = []
    upper_total = 0
    upper_hit = 0
    cand_sizes = []

    for row in rows:
        if row["lang"] != LANG:
            continue
        raw_words = row["raw"]
        norm_words = row["norm"]
        for i, raw in enumerate(raw_words):
            gold = norm_words[i] if norm_words[i] is not None else raw
            left = raw_words[i - 1] if i > 0 else "<BOS>"
            right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
            cands = gen.generate(raw)
            if gold in cands:
                upper_hit += 1
            elif force_gold:
                cands[gold].add("gold_forced")
            upper_total += 1
            cand_sizes.append(len(cands))
            for cand, sources in cands.items():
                X.append(feature_dict(raw, cand, left, right, sources, gen))
                y.append(int(cand == gold))

    upper = upper_hit / upper_total if upper_total else 0.0
    avg_cands = sum(cand_sizes) / len(cand_sizes) if cand_sizes else 0.0
    return X, np.array(y, dtype=np.int64), upper, avg_cands


def build_mfr_dictionary(rows):
    counts = defaultdict(Counter)
    for row in rows:
        if row["lang"] != LANG:
            continue
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1
    mfr = {}
    for raw, counter in counts.items():
        best, _ = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
        mfr[raw] = best
    return mfr


def evaluate_predictions(raw_sents, gold_sents, pred_sents, max_errors=80):
    total = correct = changed = changed_correct = unchanged = unchanged_correct = over_changed = 0
    errors = []
    for raw_words, gold_words, pred_words in zip(raw_sents, gold_sents, pred_sents):
        for r, g, p in zip(raw_words, gold_words, pred_words):
            total += 1
            if p == g:
                correct += 1
            if r != g:
                changed += 1
                if p == g:
                    changed_correct += 1
            else:
                unchanged += 1
                if p == g:
                    unchanged_correct += 1
                if p != r:
                    over_changed += 1
            if p != g and len(errors) < max_errors:
                errors.append((r, g, p))
    lai = (total - changed) / total if total else 0.0
    accuracy = correct / total if total else 0.0
    err = (accuracy - lai) / (1 - lai) if changed else 0.0
    changed_acc = changed_correct / changed if changed else 0.0
    unchanged_acc = unchanged_correct / unchanged if unchanged else 0.0
    over_change_rate = over_changed / unchanged if unchanged else 0.0
    return {
        "total": total,
        "changed": changed,
        "changed_rate": changed / total if total else 0.0,
        "lai": lai,
        "accuracy": accuracy,
        "err": err,
        "changed_acc": changed_acc,
        "unchanged_acc": unchanged_acc,
        "over_change_rate": over_change_rate,
        "errors": errors,
    }


def print_metrics(title, metrics, show_errors=False):
    print(f"\n[{title}]")
    print(f"Total tokens:              {metrics['total']}")
    print(f"Changed tokens:            {metrics['changed']}")
    print(f"Changed rate:              {metrics['changed_rate'] * 100:.2f}%")
    print(f"Baseline acc.(LAI):        {metrics['lai'] * 100:.2f}")
    print(f"Accuracy:                  {metrics['accuracy'] * 100:.2f}")
    print(f"ERR:                       {metrics['err'] * 100:.2f}")
    print(f"Changed token accuracy:    {metrics['changed_acc'] * 100:.2f}")
    print(f"Unchanged preservation:    {metrics['unchanged_acc'] * 100:.2f}")
    print(f"Over-change rate:          {metrics['over_change_rate'] * 100:.2f}")
    if show_errors:
        print("\nError examples: raw -> gold / pred")
        for r, g, p in metrics["errors"]:
            print(f"{r} -> {g} / {p}")


def predict_sentence(raw_words, artifact, margin: float, min_best_score: float):
    gen = artifact["generator"]
    ranker = artifact["ranker"]
    pred_words = []
    debug_counts = Counter()
    for i, raw in enumerate(raw_words):
        left = raw_words[i - 1] if i > 0 else "<BOS>"
        right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
        cands = gen.generate(raw)
        feats = []
        cand_list = []
        source_list = []
        for cand, sources in cands.items():
            cand_list.append(cand)
            source_list.append(sources)
            feats.append(feature_dict(raw, cand, left, right, sources, gen))
        probs = ranker.predict_proba(feats)[:, 1]
        best_idx = int(np.argmax(probs))
        best_cand = cand_list[best_idx]
        best_score = float(probs[best_idx])
        raw_idx = cand_list.index(raw) if raw in cand_list else -1
        original_score = float(probs[raw_idx]) if raw_idx >= 0 else 0.0
        if best_cand != raw and best_score >= min_best_score and (best_score - original_score) >= margin:
            pred_words.append(best_cand)
            debug_counts["changed"] += 1
            for s in source_list[best_idx]:
                debug_counts["source_" + s] += 1
        else:
            pred_words.append(raw)
            debug_counts["copy"] += 1
    return pred_words, debug_counts


def train(args):
    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
    train_data = train_data.filter(lambda x: x["lang"] == LANG)
    if len(train_data) == 0:
        raise ValueError(f"No IT data found in {args.train_file}")

    gen = ITCandidateGenerator(top_k_per_source=args.top_k_per_source).fit(train_data)
    X, y, upper, avg_cands = build_ranker_examples(train_data, gen, force_gold=True)
    print("train rows:", len(train_data))
    print("ranker examples:", len(y))
    print("positive examples:", int((y == 1).sum()))
    print("negative examples:", int((y == 0).sum()))
    print("candidate upperbound before gold-forcing:", f"{upper * 100:.2f}%")
    print("avg candidates/token:", f"{avg_cands:.2f}")

    if args.class_weight == "custom":
        class_weight = {0: 1.0, 1: args.positive_weight}
    elif args.class_weight == "none":
        class_weight = None
    else:
        class_weight = args.class_weight
    print("class_weight:", class_weight)

    ranker = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("clf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )),
    ])
    ranker.fit(X, y)
    train_pred = ranker.predict(X)
    print("\n[Train candidate-ranker report]")
    print(classification_report(y, train_pred, target_names=["wrong_candidate", "gold_candidate"], digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y, train_pred))

    artifact = {"lang": LANG, "generator": gen, "ranker": ranker, "params": vars(args)}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(artifact, args.output)
    print("\nsaved:", args.output)


def eval_model(args):
    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
    train_data = train_data.filter(lambda x: x["lang"] == LANG)
    valid_data = load_dataset("parquet", data_files={"valid": args.valid_file})["valid"]
    valid_data = valid_data.filter(lambda x: x["lang"] == LANG)
    raw_sents = []
    gold_sents = []
    for row in valid_data:
        raw_words = row["raw"]
        gold_words = [n if n is not None else r for r, n in zip(row["raw"], row["norm"])]
        raw_sents.append(raw_words)
        gold_sents.append(gold_words)

    mfr = build_mfr_dictionary(train_data)
    mfr_preds = [[mfr.get(w, w) for w in raw_words] for raw_words in raw_sents]
    print_metrics("MFR baseline", evaluate_predictions(raw_sents, gold_sents, mfr_preds), show_errors=False)

    artifact = joblib.load(args.model)
    gen = artifact["generator"]
    _, _, valid_upper, valid_avg_cands = build_ranker_examples(valid_data, gen, force_gold=False)
    print("\n[Candidate generation]")
    print(f"valid candidate upperbound: {valid_upper * 100:.2f}%")
    print(f"valid avg candidates/token: {valid_avg_cands:.2f}")

    pred_sents = []
    total_counts = Counter()
    for raw_words in raw_sents:
        pred_words, counts = predict_sentence(raw_words, artifact, margin=args.margin, min_best_score=args.min_best_score)
        pred_sents.append(pred_words)
        total_counts.update(counts)

    print("\nDecision counts:")
    for k, v in total_counts.most_common():
        print(f"  {k:24s} {v}")

    metrics = evaluate_predictions(raw_sents, gold_sents, pred_sents)
    print_metrics("Minimal MoNoise-style IT ranker", metrics, show_errors=args.verbose)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--train_file", type=str, default="./eval_splits_it/it_train.parquet")
    p_train.add_argument("--output", type=str, default="./models_it_monoise/it_monoise_minimal.joblib")
    p_train.add_argument("--top_k_per_source", type=int, default=8)
    p_train.add_argument("--n_estimators", type=int, default=500)
    p_train.add_argument("--max_depth", type=int, default=16)
    p_train.add_argument("--min_samples_leaf", type=int, default=1)
    p_train.add_argument("--min_samples_split", type=int, default=2)
    p_train.add_argument("--max_features", type=str, default="sqrt")
    p_train.add_argument("--class_weight", choices=["none", "custom", "balanced", "balanced_subsample"], default="balanced_subsample")
    p_train.add_argument("--positive_weight", type=float, default=5.0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--n_jobs", type=int, default=-1)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--train_file", type=str, default="./eval_splits_it/it_train.parquet")
    p_eval.add_argument("--valid_file", type=str, default="./eval_splits_it/it_valid.parquet")
    p_eval.add_argument("--model", type=str, default="./models_it_monoise/it_monoise_minimal.joblib")
    p_eval.add_argument("--margin", type=float, default=0.10)
    p_eval.add_argument("--min_best_score", type=float, default=0.20)
    p_eval.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_model(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
