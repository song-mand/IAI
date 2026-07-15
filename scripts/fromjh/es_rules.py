#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/es/es_rules.py

Spanish lexical-normalization utilities.
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict


ACCENT_MAP = {
    "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "n": "ñ",
    "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú", "N": "Ñ",
}

DEACCENT_TABLE = str.maketrans({
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
    "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ü": "U", "Ñ": "N",
})


ABBREVIATIONS = {
    "bn": ["bien"],
    "bno": ["bueno"],
    "bna": ["buena"],
    "bs": ["besos"],
    "bss": ["besos"],
    "bsss": ["besos"],
    "bsos": ["besos"],
    "bsoss": ["besos"],
    "bsits": ["besitos"],
    "cdo": ["cuando"],
    "cn": ["con"],
    "cmo": ["como"],
    "cm": ["como"],
    "d": ["de"],
    "dl": ["del"],
    "q": ["que"],
    "k": ["que"],
    "ke": ["que"],
    "qe": ["que"],
    "qé": ["qué"],
    "kiero": ["quiero"],
    "kieres": ["quieres"],
    "kiere": ["quiere"],
    "kieren": ["quieren"],
    "aki": ["aquí"],
    "akí": ["aquí"],
    "aqui": ["aquí"],
    "pk": ["porque"],
    "pq": ["porque"],
    "xq": ["porque"],
    "xk": ["porque"],
    "xke": ["porque"],
    "xque": ["porque"],
    "porq": ["porque"],
    "porqe": ["porque"],
    "sk": ["es_que"],
    "sq": ["es_que"],
    "tb": ["también"],
    "tmb": ["también"],
    "tp": ["tampoco"],
    "xfa": ["por_favor"],
    "xfavor": ["por_favor"],
    "x": ["por"],
    "pa": ["para"],
    "na": ["nada"],
    "toa": ["toda"],
    "to": ["todo"],
    "toy": ["estoy"],
    "toi": ["estoy"],
    "tas": ["estás"],
    "ta": ["está"],
    "tamos": ["estamos"],
    "dnd": ["donde"],
    "nd": ["nada"],
    "m": ["me"],
    "t": ["te"],
    "l": ["le"],
}

ACCENT_LEXICON = {
    "si": ["sí"],
    "tu": ["tú"],
    "el": ["él"],
    "mi": ["mí"],
    "mas": ["más"],
    "que": ["qué"],
    "quien": ["quién"],
    "cuando": ["cuándo"],
    "como": ["cómo"],
    "donde": ["dónde"],
    "esta": ["está", "ésta"],
    "estas": ["estás", "éstas"],
    "este": ["éste"],
    "estos": ["éstos"],
    "solo": ["sólo"],
    "se": ["sé"],
    "de": ["dé"],
    "te": ["té"],
    "aun": ["aún"],
    "ahi": ["ahí"],
    "alli": ["allí"],
    "aqui": ["aquí"],
    "dia": ["día"],
    "dias": ["días"],
    "tambien": ["también"],
    "despues": ["después"],
    "facil": ["fácil"],
    "dificil": ["difícil"],
    "ultimo": ["último"],
    "unica": ["única"],
    "republica": ["república"],
}


def target_of(raw, norm):
    return norm if norm is not None else raw


def normalize_apostrophe(s):
    return (
        s.replace("’", "'")
         .replace("`", "'")
         .replace("´", "'")
         .replace("“", '"')
         .replace("”", '"')
    )


def deaccent(s):
    return s.translate(DEACCENT_TABLE)


def strip_accents(s):
    return deaccent(s)


def collapse_repeats(s, max_repeat=2):
    return re.sub(
        r"(.)\1{" + str(max_repeat) + r",}",
        lambda m: m.group(1) * max_repeat,
        s,
    )


def full_collapse_repeats(s):
    return re.sub(r"(.)\1+", r"\1", s)


def spanish_key(s):
    s = normalize_apostrophe(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = deaccent(s)
    s = collapse_repeats(s, 2)
    return s


def is_protected_token(token):
    t = token.strip()
    if not t:
        return True
    if t.startswith("@") or t.startswith("#"):
        return True
    if t.startswith("http://") or t.startswith("https://") or t.startswith("www."):
        return True
    if re.fullmatch(r"\d+([.,:/-]\d+)*", t):
        return True
    if re.fullmatch(r"[\W_]+", t, flags=re.UNICODE):
        return True
    if re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ]{2,5}", t):
        return True
    return False


def has_long_repetition(token):
    return re.search(r"(.)\1{2,}", token.lower()) is not None


def token_shape(token):
    out = []
    for ch in token:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("0")
        elif deaccent(ch) != ch:
            out.append("á")
        else:
            out.append(ch)
    return re.sub(r"(.)\1{2,}", r"\1\1", "".join(out))


def levenshtein(a, b):
    if a is None:
        a = ""
    if b is None:
        b = ""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def accent_only_change(a, b):
    return deaccent(a).lower() == deaccent(b).lower() and a.lower() != b.lower()


def case_only_change(a, b):
    return a.lower() == b.lower() and a != b


def custom_edit_cost(raw, cand):
    if raw == cand:
        return 0.0
    r = normalize_apostrophe(raw)
    c = normalize_apostrophe(cand)
    rl = r.lower()
    cl = c.lower()

    if accent_only_change(rl, cl):
        return 0.3
    if collapse_repeats(rl, 1) == cl or collapse_repeats(rl, 2) == cl:
        return 0.3
    if spanish_key(rl) == spanish_key(cl):
        return 0.4

    transformed = rl
    low_pairs = [("k", "qu"), ("ke", "que"), ("ki", "qui"), ("x", "ch"), ("xq", "porque"), ("xk", "porque"), ("w", "gu"), ("w", "bu"), ("ao", "ado")]
    for src, dst in low_pairs:
        transformed = transformed.replace(src, dst)
    if spanish_key(transformed) == spanish_key(cl):
        return 0.6
    return float(levenshtein(spanish_key(rl), spanish_key(cl)))


def dedupe_candidates(cands):
    seen = set()
    out = []
    for source, cand in cands:
        if cand is None:
            continue
        cand = str(cand).strip()
        if not cand:
            continue
        if cand not in seen:
            seen.add(cand)
            out.append((source, cand))
    return out


def cyber_rule_candidates(raw):
    r = normalize_apostrophe(raw)
    low = r.lower()
    cands = []

    if low in ABBREVIATIONS:
        for cand in ABBREVIATIONS[low]:
            cands.append(("abbrev", cand))

    if low in ACCENT_LEXICON:
        for cand in ACCENT_LEXICON[low]:
            cands.append(("accent_lexicon", cand))

    collapsed2 = collapse_repeats(low, 2)
    collapsed1 = collapse_repeats(low, 1)
    if collapsed2 != low:
        cands.append(("repeat2", collapsed2))
    if collapsed1 != low:
        cands.append(("repeat1", collapsed1))

    rule_forms = set()
    rule_forms.add(re.sub(r"\bke\b", "que", low))
    rule_forms.add(re.sub(r"\bki", "qui", low))
    rule_forms.add(low.replace("k", "qu"))
    if "x" in low:
        rule_forms.add(low.replace("x", "ch"))
    if low.startswith("w"):
        rule_forms.add("gu" + low[1:])
        rule_forms.add("bu" + low[1:])
    if low.endswith("ao") and len(low) > 3:
        rule_forms.add(low[:-2] + "ado")
    if low.endswith("aos") and len(low) > 4:
        rule_forms.add(low[:-3] + "ados")
    if "'" in low:
        rule_forms.add(low.replace("'", ""))
        rule_forms.add(low.replace("'", " "))

    for cand in sorted(rule_forms):
        if cand and cand != low:
            cands.append(("cyber_rule", cand))

    fused = {
        "porfavor": "por_favor", "xfavor": "por_favor", "tequiero": "te_quiero",
        "tquiero": "te_quiero", "tkiero": "te_quiero", "mencanta": "me_encanta",
        "nose": "no_sé", "aver": "a_ver",
    }
    if low in fused:
        cands.append(("split_rule", fused[low]))

    return dedupe_candidates(cands)


def generate_candidates(raw, mfr=None, mfr_conf=None, key_map=None, max_key_cands=5):
    cands = [("copy", raw)]
    if mfr and raw in mfr:
        cands.append(("mfr", mfr[raw]))
    key = spanish_key(raw)
    if key_map and key in key_map:
        for cand, _count in key_map[key].most_common(max_key_cands):
            cands.append(("key", cand))
    cands.extend(cyber_rule_candidates(raw))
    if raw.lower() in ACCENT_LEXICON:
        for cand in ACCENT_LEXICON[raw.lower()]:
            cands.append(("accent", cand))
    return dedupe_candidates(cands)


def build_stats(rows, lang="es"):
    raw_counts = defaultdict(Counter)
    key_counts = defaultdict(Counter)
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()

    for row in rows:
        if row["lang"] != lang:
            continue
        norm_words = [target_of(r, n) for r, n in zip(row["raw"], row["norm"])]
        padded = ["<BOS>"] + norm_words + ["<EOS>"]
        for i, w in enumerate(norm_words):
            raw = row["raw"][i]
            raw_counts[raw][w] += 1
            key_counts[spanish_key(raw)][w] += 1
        for w in padded:
            unigram[w] += 1
        for a, b in zip(padded, padded[1:]):
            bigram[(a, b)] += 1
        for a, b, c in zip(padded, padded[1:], padded[2:]):
            trigram[(a, b, c)] += 1

    raw_stats = {}
    mfr = {}
    mfr_conf = {}
    for raw, counter in raw_counts.items():
        total = sum(counter.values())
        copy = counter.get(raw, 0)
        changed = total - copy
        best, best_count = max(counter.items(), key=lambda x: (x[1], x[0] == raw))
        raw_stats[raw] = {
            "total": total, "copy": copy, "changed": changed,
            "change_prob": changed / total if total else 0.0,
            "copy_prob": copy / total if total else 0.0,
            "best_norm": best, "best_count": best_count,
            "best_prob": best_count / total if total else 0.0,
        }
        mfr[raw] = best
        mfr_conf[raw] = best_count / total if total else 0.0

    key_stats = {}
    key_map = {}
    for key, counter in key_counts.items():
        total = sum(counter.values())
        best, best_count = counter.most_common(1)[0]
        key_stats[key] = {"total": total, "best_norm": best, "best_count": best_count, "best_prob": best_count / total if total else 0.0}
        key_map[key] = counter

    return {"raw_stats": raw_stats, "key_stats": key_stats, "mfr": mfr, "mfr_conf": mfr_conf, "key_map": key_map, "unigram": unigram, "bigram": bigram, "trigram": trigram}


def detector_features(raw, left, right, resources):
    raw_stats = resources["raw_stats"]
    key_stats = resources["key_stats"]
    low = raw.lower()
    key = spanish_key(raw)
    letters = sum(ch.isalpha() for ch in raw)
    digits = sum(ch.isdigit() for ch in raw)
    punct = sum((not ch.isalnum()) for ch in raw)
    rs = raw_stats.get(raw)
    ks = key_stats.get(key)
    return {
        "bias": 1,
        "raw_lower=" + low: 1,
        "key=" + key: 1,
        "shape=" + token_shape(raw): 1,
        "left_key=" + spanish_key(left): 1,
        "right_key=" + spanish_key(right): 1,
        "prefix1=" + key[:1]: 1,
        "prefix2=" + key[:2]: 1,
        "prefix3=" + key[:3]: 1,
        "suffix1=" + key[-1:]: 1,
        "suffix2=" + key[-2:]: 1,
        "suffix3=" + key[-3:]: 1,
        "len": min(len(raw), 30),
        "letters": min(letters, 30),
        "digits": min(digits, 30),
        "punct": min(punct, 30),
        "is_protected": int(is_protected_token(raw)),
        "has_long_repetition": int(has_long_repetition(raw)),
        "has_accent": int(deaccent(raw) != raw),
        "is_all_lower": int(raw.islower()),
        "is_all_upper": int(raw.isupper()),
        "has_kqxw": int(any(ch in low for ch in ["k", "q", "x", "w"])),
        "raw_seen": int(rs is not None),
        "raw_total": min(rs["total"], 10) if rs else 0,
        "raw_change_prob": rs["change_prob"] if rs else 0.0,
        "raw_copy_prob": rs["copy_prob"] if rs else 0.0,
        "raw_best_is_copy": int(rs["best_norm"] == raw) if rs else 0,
        "raw_best_prob": rs["best_prob"] if rs else 0.0,
        "key_seen": int(ks is not None),
        "key_total": min(ks["total"], 10) if ks else 0,
        "key_best_prob": ks["best_prob"] if ks else 0.0,
        "key_best_is_raw": int(ks["best_norm"] == raw) if ks else 0,
    }


def ngram_scores(left_norm, cand, right_norm, resources):
    unigram = resources["unigram"]
    bigram = resources["bigram"]
    trigram = resources["trigram"]
    vocab = max(len(unigram), 1)
    total = sum(unigram.values()) + vocab
    uni = math.log((unigram.get(cand, 0) + 1) / total)
    left_total = unigram.get(left_norm, 0) + vocab
    bi_l = math.log((bigram.get((left_norm, cand), 0) + 1) / left_total)
    cand_total = unigram.get(cand, 0) + vocab
    bi_r = math.log((bigram.get((cand, right_norm), 0) + 1) / cand_total)
    tri_total = bigram.get((left_norm, cand), 0) + vocab
    tri = math.log((trigram.get((left_norm, cand, right_norm), 0) + 1) / tri_total)
    return uni, bi_l, bi_r, tri


def candidate_features(raw, cand, source, left, right, resources):
    mfr = resources["mfr"]
    mfr_conf = resources["mfr_conf"]
    key = spanish_key(raw)
    left_norm = mfr.get(left, left)
    right_norm = mfr.get(right, right)
    uni, bi_l, bi_r, tri = ngram_scores(left_norm, cand, right_norm, resources)
    return {
        "bias": 1,
        "source=" + source: 1,
        "raw_lower=" + raw.lower(): 1,
        "cand_lower=" + cand.lower(): 1,
        "raw_key=" + key: 1,
        "cand_key=" + spanish_key(cand): 1,
        "shape_raw=" + token_shape(raw): 1,
        "shape_cand=" + token_shape(cand): 1,
        "candidate_is_copy": int(cand == raw),
        "candidate_is_mfr": int(cand == mfr.get(raw, raw)),
        "mfr_conf": mfr_conf.get(raw, 0.0),
        "levenshtein": min(levenshtein(raw.lower(), cand.lower()), 10),
        "custom_edit_cost": min(custom_edit_cost(raw, cand), 10),
        "accent_only": int(accent_only_change(raw, cand)),
        "case_only": int(case_only_change(raw, cand)),
        "same_key": int(spanish_key(raw) == spanish_key(cand)),
        "raw_len": min(len(raw), 30),
        "cand_len": min(len(cand), 30),
        "len_delta_abs": min(abs(len(cand) - len(raw)), 20),
        "cand_has_space_or_us": int((" " in cand) or ("_" in cand)),
        "left_key=" + spanish_key(left): 1,
        "right_key=" + spanish_key(right): 1,
        "lm_uni": uni,
        "lm_bi_left": bi_l,
        "lm_bi_right": bi_r,
        "lm_tri": tri,
    }


def safe_byt5_output(raw, pred, allow_underscore=True):
    if pred is None:
        return None
    p = str(pred).strip()
    if not p:
        return None
    low = p.lower()
    bad = ["lang:", "word:", "context:", "target:", "<extra_id", "extra_id"]
    if any(m in low for m in bad):
        return None
    if len(p) > max(20, len(raw) * 3):
        return None
    if " " in p:
        return None
    if "_" in p and not allow_underscore:
        return None
    if custom_edit_cost(raw, p) > 4.0 and spanish_key(raw) != spanish_key(p):
        return None
    return p
