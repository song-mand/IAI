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
from tqdm import tqdm

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

    # accent / apostrophe frequent forms
    "e'": ["è"],
    "E'": ["È"],
    "perche'": ["perché"],
    "perchè": ["perché"],
    "piu'": ["più"],
    "li'": ["lì"],
    "po": ["po'"],
    "un": ["un'"],

    # common 1-N candidate examples; candidate remains one string in the norm list.
    "cé": ["c' è"],
    "cè": ["c' è"],
}


def strip_accents(s: str) -> str:
    """Remove all Unicode combining marks.

    This handles not only Italian accents such as è/é/à,
    but also tone-mark-like combining marks used in other Latin scripts.
    Example: tiếng -> tieng, không -> khong.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def count_diacritics(s: str) -> int:
    """Count Unicode combining marks after decomposition."""
    return sum(
        1 for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) == "Mn"
    )


def has_diacritic(s: str) -> bool:
    return count_diacritics(s) > 0


def diacritic_key(s: str) -> str:
    """Key that ignores accents/diacritics/tone marks but keeps base letters.

    This is broader than Italian-only accent handling and lets the generator
    connect forms like realta/realtà, Unitá/Unità, and also tone-marked
    Latin-script forms if they appear in multilingual data.
    """
    s = unicodedata.normalize("NFKC", normalize_apostrophe(s))
    s = strip_accents(s)
    s = s.lower()
    return collapse_repeats(s, max_repeat=2)


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
    """Extended MoNoise-style candidate generator for Italian.

    Candidate modules:
      1. original token
      2. train lookup raw -> observed norm / MFR
      3. normalized-key lookup
      4. case candidate from train norm vocabulary
      5. abbreviation/accent/apostrophe rule candidates
      6. Unicode diacritic/tone-mark candidates from train norm vocabulary
      7. repeat-collapse candidates
      8. split candidates

    Optional ByT5 candidates are added outside this class because they depend on sentence context.
    """

    def __init__(self, top_k_per_source: int = 8, max_split_candidates: int = 8):
        self.top_k_per_source = top_k_per_source
        self.max_split_candidates = max_split_candidates
        self.exact = defaultdict(Counter)
        self.key_index = defaultdict(Counter)
        self.case_index = defaultdict(Counter)
        self.diacritic_index = defaultdict(Counter)
        self.norm_vocab = Counter()
        self.unigram = Counter()
        self.bigram_left = Counter()
        self.bigram_right = Counter()

    def fit(self, rows):
        for row in rows:
            if row["lang"] != LANG:
                continue
            raw_words = row["raw"]
            norm_words = [n if n is not None else r for r, n in zip(row["raw"], row["norm"])]
            for i, (raw, target) in enumerate(zip(raw_words, norm_words)):
                self.exact[raw][target] += 1
                self.key_index[make_key(raw)][target] += 1
                self.case_index[target.lower()][target] += 1
                self.diacritic_index[diacritic_key(target)][target] += 1
                self.norm_vocab[target] += 1
                left = raw_words[i - 1] if i > 0 else "<BOS>"
                right = raw_words[i + 1] if i + 1 < len(raw_words) else "<EOS>"
                self.unigram[target] += 1
                self.bigram_left[(left.lower(), target)] += 1
                self.bigram_right[(target, right.lower())] += 1
        return self

    def _add(self, table, cand, source):
        if cand is None:
            return
        cand = str(cand).strip()
        if cand == "":
            return
        table[cand].add(source)

    def _add_counter_top(self, table, counter, source, top_k=None):
        if top_k is None:
            top_k = self.top_k_per_source
        for cand, _ in counter.most_common(top_k):
            self._add(table, cand, source)

    def add_diacritic_candidates(self, raw: str, candidates):
        # Generic Unicode diacritic/tone-mark module.
        # It generates candidates that share the same base letters after removing
        # combining marks. This covers Italian accents and Latin-script tone marks.
        key = diacritic_key(raw)
        if key in self.diacritic_index:
            for cand, _ in self.diacritic_index[key].most_common(self.top_k_per_source):
                if cand != raw:
                    self._add(candidates, cand, "diacritic")

    def add_repeat_candidates(self, raw: str, candidates):
        # Collapse 3+ repeated chars to 1 or 2 chars. Examples:
        # mondoooo -> mondo, parteeeeeeee -> parte
        for keep in (1, 2):
            collapsed = collapse_repeats(raw, max_repeat=keep)
            if collapsed != raw:
                if collapsed in self.norm_vocab:
                    self._add(candidates, collapsed, "repeat")
                # Also allow the most frequent case form of the collapsed candidate.
                low = collapsed.lower()
                if low in self.case_index:
                    self._add_counter_top(candidates, self.case_index[low], "repeat_case", top_k=3)
                # And candidates sharing normalized key.
                key = make_key(collapsed)
                if key in self.key_index:
                    self._add_counter_top(candidates, self.key_index[key], "repeat_key", top_k=3)
                # Also use diacritic/tone-insensitive norm-vocabulary index.
                dkey = diacritic_key(collapsed)
                if dkey in self.diacritic_index:
                    self._add_counter_top(candidates, self.diacritic_index[dkey], "repeat_diacritic", top_k=3)

    def add_split_candidates(self, raw: str, candidates):
        # MoNoise-style split module: split on every position and keep splits whose parts
        # are observed canonical norm tokens. Candidate remains one string, e.g. "c' è".
        if len(raw) <= 3 or is_protected_token(raw):
            return

        added = 0
        for pos in range(1, len(raw)):
            left = raw[:pos]
            right = raw[pos:]

            variants = [(left, right)]

            # Useful IT apostrophe variants, e.g. ce -> c' è is mainly handled by rules,
            # but this enables things like lho -> l' ho if both parts are in vocab.
            if len(left) == 1:
                variants.append((left + "'", right))

            for lpart, rpart in variants:
                if lpart in self.norm_vocab and rpart in self.norm_vocab:
                    self._add(candidates, f"{lpart} {rpart}", "split")
                    added += 1
                    if added >= self.max_split_candidates:
                        return

                # Case-insensitive fallback through case_index.
                l_counter = self.case_index.get(lpart.lower())
                r_counter = self.case_index.get(rpart.lower())
                if l_counter and r_counter:
                    lbest, _ = l_counter.most_common(1)[0]
                    rbest, _ = r_counter.most_common(1)[0]
                    self._add(candidates, f"{lbest} {rbest}", "split_case")
                    added += 1
                    if added >= self.max_split_candidates:
                        return

                # Diacritic/tone-insensitive fallback for split parts.
                ld_counter = self.diacritic_index.get(diacritic_key(lpart))
                rd_counter = self.diacritic_index.get(diacritic_key(rpart))
                if ld_counter and rd_counter:
                    lbest, _ = ld_counter.most_common(1)[0]
                    rbest, _ = rd_counter.most_common(1)[0]
                    self._add(candidates, f"{lbest} {rbest}", "split_diacritic")
                    added += 1
                    if added >= self.max_split_candidates:
                        return

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

        self.add_diacritic_candidates(raw, candidates)
        self.add_repeat_candidates(raw, candidates)
        self.add_split_candidates(raw, candidates)

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

    diacritic_counter = gen.diacritic_index.get(diacritic_key(raw), Counter())
    diacritic_total = sum(diacritic_counter.values())
    diacritic_count = diacritic_counter.get(cand, 0)
    diacritic_conf = diacritic_count / diacritic_total if diacritic_total else 0.0

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
        "diacritic_count": diacritic_count,
        "diacritic_conf": diacritic_conf,
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
        "from_diacritic": int("diacritic" in sources),
        "from_repeat": int("repeat" in sources or "repeat_case" in sources or "repeat_key" in sources or "repeat_diacritic" in sources),
        "from_split": int("split" in sources or "split_case" in sources or "split_diacritic" in sources),
        "from_byt5": int("byt5" in sources),
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
        "diacritic_count_log": math.log1p(stats["diacritic_count"]),
        "diacritic_conf": stats["diacritic_conf"],

        "cand_freq_log": math.log1p(gen.norm_vocab.get(cand, 0)),
        "unigram_log": math.log1p(gen.unigram.get(cand, 0)),
        "left_bigram_log": math.log1p(gen.bigram_left.get((left.lower(), cand), 0)),
        "right_bigram_log": math.log1p(gen.bigram_right.get((cand, right.lower()), 0)),

        "same_string": int(raw == cand),
        "same_lower": int(raw.lower() == cand.lower()),
        "same_key": int(raw_key == cand_key),
        "same_accentless": int(strip_accents(raw).lower() == strip_accents(cand).lower()),
        "same_diacritic_key": int(diacritic_key(raw) == diacritic_key(cand)),
        "edit_distance": min(dist, 10),
        "norm_edit_distance": dist / max_len,
        "len_raw": min(len(raw), 40),
        "len_cand": min(len(cand), 40),
        "len_diff": min(abs(len(raw) - len(cand)), 40),
        "cand_has_space": int(" " in cand),
        "chars_order_preserved": int(char_order_preserved(raw.replace(" ", ""), cand.replace(" ", ""))),
        "contains_alpha_raw": int(contains_alpha(raw)),
        "contains_alpha_cand": int(contains_alpha(cand)),

        "raw_has_accent": int(strip_accents(raw) != raw),
        "cand_has_accent": int(strip_accents(cand) != cand),
        "raw_diacritic_count": min(count_diacritics(raw), 8),
        "cand_diacritic_count": min(count_diacritics(cand), 8),
        "diacritic_count_diff": min(abs(count_diacritics(raw) - count_diacritics(cand)), 8),
        "diacritic_added": int(count_diacritics(raw) == 0 and count_diacritics(cand) > 0),
        "diacritic_removed": int(count_diacritics(raw) > 0 and count_diacritics(cand) == 0),
        "diacritic_changed": int(strip_accents(raw).lower() == strip_accents(cand).lower() and raw != cand),
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
    upper_total = upper_hit = 0
    changed_total = changed_hit = 0
    unchanged_total = unchanged_hit = 0
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

            hit = gold in cands
            upper_total += 1
            upper_hit += int(hit)
            if raw != gold:
                changed_total += 1
                changed_hit += int(hit)
            else:
                unchanged_total += 1
                unchanged_hit += int(hit)

            if not hit and force_gold:
                cands[gold].add("gold_forced")

            cand_sizes.append(len(cands))
            for cand, sources in cands.items():
                X.append(feature_dict(raw, cand, left, right, sources, gen))
                y.append(int(cand == gold))

    stats = {
        "upper": upper_hit / upper_total if upper_total else 0.0,
        "changed_upper": changed_hit / changed_total if changed_total else 0.0,
        "unchanged_upper": unchanged_hit / unchanged_total if unchanged_total else 0.0,
        "avg_cands": sum(cand_sizes) / len(cand_sizes) if cand_sizes else 0.0,
        "tokens": upper_total,
        "changed_tokens": changed_total,
    }
    return X, np.array(y, dtype=np.int64), stats


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

def analyze_gold_in_candidates(raw_sents, gold_sents, pred_sents, artifact, max_examples=120):
    """
    현재 candidate generator가 만든 후보 안에 gold 정답이 들어있는지 확인한다.

    목적:
      - 틀린 이유가 candidate generation 문제인지 확인
      - gold가 candidates 안에 없으면 ranker는 절대 맞출 수 없음
      - gold가 candidates 안에 있는데 틀렸으면 ranker/selection 문제
    """
    gen = artifact["generator"]

    changed_total = 0
    changed_gold_in = 0
    unchanged_total = 0
    unchanged_gold_in = 0

    error_total = 0
    error_gold_in = 0
    error_gold_missing = 0

    changed_missing_examples = []
    error_examples = []

    for sent_idx, (raw_words, gold_words, pred_words) in enumerate(zip(raw_sents, gold_sents, pred_sents)):
        for tok_idx, (raw, gold, pred) in enumerate(zip(raw_words, gold_words, pred_words)):
            candidates = gen.generate(raw)
            gold_in_candidates = gold in candidates

            if raw != gold:
                changed_total += 1
                if gold_in_candidates:
                    changed_gold_in += 1
                else:
                    if len(changed_missing_examples) < max_examples:
                        cand_list = [
                            f"{cand}<{','.join(sorted(sources))}>"
                            for cand, sources in candidates.items()
                        ]
                        changed_missing_examples.append(
                            (raw, gold, pred, cand_list)
                        )
            else:
                unchanged_total += 1
                if gold_in_candidates:
                    unchanged_gold_in += 1

            if pred != gold:
                error_total += 1
                if gold_in_candidates:
                    error_gold_in += 1
                else:
                    error_gold_missing += 1

                if len(error_examples) < max_examples:
                    cand_list = [
                        f"{cand}<{','.join(sorted(sources))}>"
                        for cand, sources in candidates.items()
                    ]

                    gold_sources = sorted(candidates[gold]) if gold_in_candidates else []
                    pred_sources = sorted(candidates[pred]) if pred in candidates else []

                    error_examples.append({
                        "raw": raw,
                        "gold": gold,
                        "pred": pred,
                        "gold_in_candidates": gold_in_candidates,
                        "gold_sources": gold_sources,
                        "pred_in_candidates": pred in candidates,
                        "pred_sources": pred_sources,
                        "candidates": cand_list,
                    })

    changed_upper = changed_gold_in / changed_total if changed_total else 0.0
    unchanged_upper = unchanged_gold_in / unchanged_total if unchanged_total else 0.0
    error_gold_in_rate = error_gold_in / error_total if error_total else 0.0

    print("\n[Gold-in-candidates analysis]")
    print(f"Changed tokens:                  {changed_total}")
    print(f"Changed gold in candidates:       {changed_gold_in}")
    print(f"Changed gold missing:             {changed_total - changed_gold_in}")
    print(f"Changed candidate upperbound:     {changed_upper * 100:.2f}%")
    print()
    print(f"Unchanged tokens:                {unchanged_total}")
    print(f"Unchanged gold in candidates:     {unchanged_gold_in}")
    print(f"Unchanged candidate upperbound:   {unchanged_upper * 100:.2f}%")
    print()
    print(f"Prediction errors:                {error_total}")
    print(f"Errors where gold is candidate:   {error_gold_in}")
    print(f"Errors where gold is missing:     {error_gold_missing}")
    print(f"Error gold-in-candidates rate:    {error_gold_in_rate * 100:.2f}%")

    print("\n[Changed tokens where gold is NOT in candidates]")
    if not changed_missing_examples:
        print("None")
    else:
        for raw, gold, pred, cand_list in changed_missing_examples:
            print(f"raw={raw} gold={gold} pred={pred}")
            print(f"  candidates={cand_list}")

    print("\n[Error examples with gold_in_candidates]")
    if not error_examples:
        print("None")
    else:
        for ex in error_examples:
            print(
                f"raw={ex['raw']} gold={ex['gold']} pred={ex['pred']} "
                f"gold_in_candidates={ex['gold_in_candidates']}"
            )
            print(f"  gold_sources={ex['gold_sources']}")
            print(f"  pred_in_candidates={ex['pred_in_candidates']}")
            print(f"  pred_sources={ex['pred_sources']}")
            print(f"  candidates={ex['candidates']}")

def train(args):
    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
    train_data = train_data.filter(lambda x: x["lang"] == LANG)
    if len(train_data) == 0:
        raise ValueError(f"No IT data found in {args.train_file}")

    gen = ITCandidateGenerator(
        top_k_per_source=args.top_k_per_source,
        max_split_candidates=args.max_split_candidates,
    ).fit(train_data)

    X, y, stats = build_ranker_examples(train_data, gen, force_gold=True)
    print("train rows:", len(train_data))
    print("ranker examples:", len(y))
    print("positive examples:", int((y == 1).sum()))
    print("negative examples:", int((y == 0).sum()))
    print("candidate upperbound before gold-forcing:", f"{stats['upper'] * 100:.2f}%")
    print("changed candidate upperbound before gold-forcing:", f"{stats['changed_upper'] * 100:.2f}%")
    print("unchanged candidate upperbound before gold-forcing:", f"{stats['unchanged_upper'] * 100:.2f}%")
    print("avg candidates/token:", f"{stats['avg_cands']:.2f}")

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


def load_byt5(model_path, device):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception as e:
        raise ImportError("ByT5 candidate requires torch and transformers") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def build_sentinel_input(raw_words, i):
    words_copy = list(raw_words)
    words_copy[i] = f"<extra_id_0> {raw_words[i]} <extra_id_1>"
    return " ".join(words_copy)


def batch_byt5_candidates(raw_sents, model_path, mode="missing", batch_size=32, max_input_len=128, num_beams=2):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("ByT5 candidate device:", device)
    tokenizer, model = load_byt5(model_path, device)

    jobs = []
    for si, raw_words in enumerate(raw_sents):
        for ti, raw in enumerate(raw_words):
            if is_protected_token(raw):
                continue
            # mode is filtered later because candidate availability is known there.
            jobs.append((si, ti, build_sentinel_input(raw_words, ti)))

    out = {}
    for start in tqdm(range(0, len(jobs), batch_size), desc="ByT5 candidates"):
        batch = jobs[start:start + batch_size]
        inputs = tokenizer(
            [x[2] for x in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=num_beams)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for (si, ti, _), pred in zip(batch, preds):
            pred = pred.strip()
            if pred:
                out[(si, ti)] = pred

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def reject_byt5_candidate(raw, pred):
    if not pred or pred.strip() == "":
        return True
    pred = pred.strip()
    if is_protected_token(raw) and pred != raw:
        return True
    if pred.startswith("@") or pred.startswith("#") or "http" in pred or "t.co" in pred:
        return True
    if len(pred) > max(len(raw) * 3, 20):
        return True
    return False


def build_all_prediction_instances(raw_sents, artifact, byt5_map=None, byt5_mode="missing"):
    gen = artifact["generator"]
    all_features = []
    groups = []

    for si, raw_words in enumerate(tqdm(raw_sents, desc="build candidates")):
        for ti, raw in enumerate(raw_words):
            left = raw_words[ti - 1] if ti > 0 else "<BOS>"
            right = raw_words[ti + 1] if ti + 1 < len(raw_words) else "<EOS>"
            cands = gen.generate(raw)

            # Optional ByT5 candidate as a low-priority candidate source.
            if byt5_map is not None and not is_protected_token(raw):
                non_original = [c for c in cands if c != raw]
                should_add = byt5_mode == "all" or (byt5_mode == "missing" and len(non_original) == 0)
                if should_add:
                    pred = byt5_map.get((si, ti))
                    if pred and not reject_byt5_candidate(raw, pred):
                        cands[pred].add("byt5")

            start = len(all_features)
            cand_list = []
            source_list = []
            for cand, sources in cands.items():
                cand_list.append(cand)
                source_list.append(sources)
                all_features.append(feature_dict(raw, cand, left, right, sources, gen))
            end = len(all_features)
            groups.append((si, ti, raw, start, end, cand_list, source_list))
    return all_features, groups


def predict_all(raw_sents, artifact, margin, min_best_score, byt5_map=None, byt5_mode="missing"):
    ranker = artifact["ranker"]
    all_features, groups = build_all_prediction_instances(raw_sents, artifact, byt5_map=byt5_map, byt5_mode=byt5_mode)
    print("total candidate instances:", len(all_features))
    probs_all = ranker.predict_proba(all_features)[:, 1]

    pred_sents = [list(sent) for sent in raw_sents]
    counts = Counter()
    for si, ti, raw, start, end, cand_list, source_list in tqdm(groups, desc="select predictions"):
        probs = probs_all[start:end]
        best_i = int(np.argmax(probs))
        best_cand = cand_list[best_i]
        best_score = float(probs[best_i])
        raw_i = cand_list.index(raw) if raw in cand_list else -1
        original_score = float(probs[raw_i]) if raw_i >= 0 else 0.0
        if best_cand != raw and best_score >= min_best_score and (best_score - original_score) >= margin:
            pred_sents[si][ti] = best_cand
            counts["changed"] += 1
            for src in source_list[best_i]:
                counts["source_" + src] += 1
        else:
            pred_sents[si][ti] = raw
            counts["copy"] += 1
    return pred_sents, counts


def eval_model(args):
    train_data = load_dataset("parquet", data_files={"train": args.train_file})["train"]
    train_data = train_data.filter(lambda x: x["lang"] == LANG)
    valid_data = load_dataset("parquet", data_files={"valid": args.valid_file})["valid"]
    valid_data = valid_data.filter(lambda x: x["lang"] == LANG)

    raw_sents, gold_sents = [], []
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
    _, _, valid_stats = build_ranker_examples(valid_data, gen, force_gold=False)
    print("\n[Candidate generation without optional ByT5]")
    print(f"valid candidate upperbound: {valid_stats['upper'] * 100:.2f}%")
    print(f"valid changed candidate upperbound: {valid_stats['changed_upper'] * 100:.2f}%")
    print(f"valid unchanged candidate upperbound: {valid_stats['unchanged_upper'] * 100:.2f}%")
    print(f"valid avg candidates/token: {valid_stats['avg_cands']:.2f}")

    byt5_map = None
    if args.use_byt5_candidate:
        if not os.path.exists(args.byt5_model_path):
            raise FileNotFoundError(f"ByT5 model not found: {args.byt5_model_path}")
        byt5_map = batch_byt5_candidates(
            raw_sents,
            args.byt5_model_path,
            mode=args.byt5_mode,
            batch_size=args.byt5_batch_size,
            max_input_len=args.max_input_len,
            num_beams=args.num_beams,
        )

    pred_sents, counts = predict_all(
        raw_sents,
        artifact,
        margin=args.margin,
        min_best_score=args.min_best_score,
        byt5_map=byt5_map,
        byt5_mode=args.byt5_mode,
    )

    print("\nDecision counts:")
    for k, v in counts.most_common():
        print(f"  {k:24s} {v}")

    if args.show_gold_candidates:
        analyze_gold_in_candidates(
            raw_sents=raw_sents,
            gold_sents=gold_sents,
            pred_sents=pred_sents,
            artifact=artifact,
            max_examples=args.max_gold_candidate_examples,
        )

    metrics = evaluate_predictions(raw_sents, gold_sents, pred_sents)
    title = "Extended MoNoise-style IT ranker"
    if args.use_byt5_candidate:
        title += " + optional ByT5 candidate"
    print_metrics(title, metrics, show_errors=args.verbose)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train_file", type=str, default="./eval_splits_it_ext/it_train.parquet")
    p_train.add_argument("--output", type=str, default="./models_it_monoise_extend_candidates/it_monoise_extend_candidates.joblib")
    p_train.add_argument("--top_k_per_source", type=int, default=8)
    p_train.add_argument("--max_split_candidates", type=int, default=8)
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
    p_eval.add_argument("--train_file", type=str, default="./eval_splits_it_ext/it_train.parquet")
    p_eval.add_argument("--valid_file", type=str, default="./eval_splits_it_ext/it_valid.parquet")
    p_eval.add_argument("--model", type=str, default="./models_it_monoise_extend_candidates/it_monoise_extend_candidates.joblib")
    p_eval.add_argument("--margin", type=float, default=0.10)
    p_eval.add_argument("--min_best_score", type=float, default=0.20)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.add_argument("--show_gold_candidates", action="store_true")
    p_eval.add_argument("--max_gold_candidate_examples", type=int, default=80)

    # Optional ByT5 candidate source. Off by default.
    p_eval.add_argument("--use_byt5_candidate", action="store_true")
    p_eval.add_argument("--byt5_model_path", type=str, default="./final_model_eval_it/it_model")
    p_eval.add_argument("--byt5_mode", choices=["missing", "all"], default="missing")
    p_eval.add_argument("--byt5_batch_size", type=int, default=32)
    p_eval.add_argument("--max_input_len", type=int, default=128)
    p_eval.add_argument("--num_beams", type=int, default=2)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_model(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
