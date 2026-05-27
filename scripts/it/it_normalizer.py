import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


DEFAULT_CONFIG = {
    "use_exact_mfr": True,
    "use_casefold_mfr": True,
    "use_diacritics": True,
    "use_safe_abbrev": True,
    "use_context_abbrev": True,
    "use_repeat": True,
    "use_capitalization": False,
    "use_split": False,

    "min_count_exact": 1,
    "min_conf_exact": 0.55,
    "min_count_casefold": 2,
    "min_conf_casefold": 0.70,

    "decision_margin": 1.15,
    "standard_raw_penalty": 1.25,
    "named_entity_penalty": 1.25,
    "multiword_penalty": 2.50,
}


URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MENTION_RE = re.compile(r"^@\w+")
HASHTAG_RE = re.compile(r"^#\S+")
PUNCT_ONLY_RE = re.compile(r"^[^\wÀ-ÖØ-öø-ÿ]+$")
LAUGH_RE = re.compile(r"^(a?ha)+h?$|^(e?he)+h?$|^x+d+$", re.IGNORECASE)
REPEAT_RE = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ])\1{2,}")

PHRASAL_OR_NONWORD_KEEP = {
    "lol", "omg", "rofl", "lmao", "tvb", "tvtb",
    "ahah", "ahaha", "ahahah", "eheh", "hehe",
    "boh", "mah", "beh", "bhe",
}

KNOWN_ACRONYM_SEED = {
    "PD", "FI", "M5S", "UE", "USA", "ONU", "CISL", "CGIL", "UIL",
    "TV", "SMS", "NLP", "COVID",
}

SAFE_ABBREV = {
    "cmq": "comunque",
    "cqm": "comunque",
    "nn": "non",
    "nnt": "niente",
    "nnt.": "niente",
    "ke": "che",
    "k": "che",
    "ki": "chi",
    "xke": "perché",
    "xké": "perché",
    "xkè": "perché",
    "xche": "perché",
    "xchè": "perché",
    "xché": "perché",
    "perke": "perché",
    "perké": "perché",
    "perkè": "perché",
    "xò": "però",
    "xo": "però",
    "grz": "grazie",
    "qnd": "quando",
    "qlc": "qualche",
    "qlcs": "qualcosa",
    "qlcn": "qualcuno",
    "dv": "dove",
    "cn": "con",
}

CONTEXT_ABBREV = {
    "x": ["per"],
    "6": ["sei"],
    "c": ["ci"],
    "sn": ["sono"],
    "qst": ["questo", "questa", "questi", "queste"],
    "qsto": ["questo"],
    "qsta": ["questa"],
    "qsti": ["questi"],
    "qste": ["queste"],
    "tt": ["tutto", "tutta", "tutti", "tutte"],
    "tutti": ["tutti"],
}

DIACRITIC_EXACT = {
    "e'": "è",
    "e`": "è",
    "e´": "è",
    "E'": "È",
    "E`": "È",
    "E´": "È",
    "c'e": "c'è",
    "c'e'": "c'è",
    "c`e": "c'è",
    "dov'e": "dov'è",
    "dov'e'": "dov'è",
}

DIACRITIC_LOWER = {
    "perche": "perché",
    "perche'": "perché",
    "perchè": "perché",
    "perké": "perché",
    "piu": "più",
    "piu'": "più",
    "puo": "può",
    "puo'": "può",
    "cosi": "così",
    "cosi'": "così",
    "pero": "però",
    "pero'": "però",
    "gia": "già",
    "gia'": "già",
    "citta": "città",
    "citta'": "città",
}

APOSTROPHE_PROTECTED = {
    "po'", "po’", "un'", "un’", "l'", "l’", "dell'", "dell’",
    "all'", "all’", "nell'", "nell’", "sull'", "sull’",
}

SPLIT_CANDIDATES = {
    "vabbene": "va bene",
    "vabene": "va bene",
    "apposto": "a posto",
    "apparte": "a parte",
}


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(override or {})
    return out


def normalize_case_like(raw: str, cand: str) -> str:
    if not raw:
        return cand
    if raw.isupper() and len(raw) > 1:
        return cand.upper()
    if raw[0].isupper():
        return cand[:1].upper() + cand[1:]
    return cand


def is_punct_only(token: str) -> bool:
    return bool(PUNCT_ONLY_RE.match(token))


def is_url(token: str) -> bool:
    return bool(URL_RE.match(token))


def is_email(token: str) -> bool:
    return bool(EMAIL_RE.match(token))


def is_mention(token: str) -> bool:
    return bool(MENTION_RE.match(token))


def is_hashtag(token: str) -> bool:
    return bool(HASHTAG_RE.match(token))


def looks_like_acronym(token: str, known_acronyms: set) -> bool:
    if token in known_acronyms:
        return True
    if len(token) >= 2 and token.isupper() and any(ch.isalpha() for ch in token):
        return True
    return False


def looks_like_named_entity(token: str) -> bool:
    if len(token) <= 1:
        return False
    if token[0].isupper() and not token.isupper():
        return True
    return False


def is_protected_token(token: str, known_acronyms: set) -> bool:
    low = token.lower()

    if not token:
        return True
    if is_url(token) or is_email(token):
        return True
    if is_mention(token) or is_hashtag(token):
        return True
    if is_punct_only(token):
        return True
    if low in PHRASAL_OR_NONWORD_KEEP:
        return True
    if LAUGH_RE.match(low):
        return True
    if looks_like_acronym(token, known_acronyms):
        return True

    return False


def best_mapping(counter: Counter, raw: str, prefer_identity: bool = True) -> Tuple[str, int, float, int]:
    total = sum(counter.values())
    max_count = max(counter.values())
    bests = [k for k, v in counter.items() if v == max_count]

    if prefer_identity and raw in bests:
        best = raw
    else:
        best = sorted(bests)[0]

    conf = max_count / total if total else 0.0
    return best, max_count, conf, total


def build_artifacts_from_rows(rows: List[Dict[str, Any]], prefer_identity: bool = True) -> Dict[str, Any]:
    exact_counts = defaultdict(Counter)
    casefold_counts = defaultdict(Counter)
    norm_vocab = Counter()
    raw_vocab = Counter()
    acronym_counts = Counter()

    for row in rows:
        raw_words = row["raw"]
        norm_words = row["norm"]

        for raw, norm in zip(raw_words, norm_words):
            if norm is None:
                norm = raw

            raw = str(raw)
            norm = str(norm)

            exact_counts[raw][norm] += 1
            casefold_counts[raw.lower()][norm] += 1
            raw_vocab[raw] += 1
            norm_vocab[norm] += 1

            if raw == norm and raw.isupper() and len(raw) >= 2:
                acronym_counts[raw] += 1

    exact = {}
    for raw, counter in exact_counts.items():
        norm, count, conf, total = best_mapping(counter, raw, prefer_identity=prefer_identity)
        exact[raw] = {
            "norm": norm,
            "count": count,
            "total": total,
            "conf": conf,
            "changed": norm != raw,
        }

    casefold = {}
    for raw_low, counter in casefold_counts.items():
        norm, count, conf, total = best_mapping(counter, raw_low, prefer_identity=False)
        casefold[raw_low] = {
            "norm": norm,
            "count": count,
            "total": total,
            "conf": conf,
            "changed": norm != raw_low,
        }

    known_acronyms = set(KNOWN_ACRONYM_SEED)
    known_acronyms.update([k for k, v in acronym_counts.items() if v >= 2])

    return {
        "exact": exact,
        "casefold": casefold,
        "norm_vocab": dict(norm_vocab),
        "raw_vocab": dict(raw_vocab),
        "known_acronyms": sorted(known_acronyms),
        "config": dict(DEFAULT_CONFIG),
    }


class ITConservativeNormalizer:
    def __init__(self, artifacts: Dict[str, Any], config: Dict[str, Any] = None):
        self.artifacts = artifacts
        self.config = merge_config(DEFAULT_CONFIG, artifacts.get("config", {}))
        self.config = merge_config(self.config, config or {})

        self.exact = artifacts.get("exact", {})
        self.casefold = artifacts.get("casefold", {})
        self.norm_vocab = artifacts.get("norm_vocab", {})
        self.raw_vocab = artifacts.get("raw_vocab", {})
        self.known_acronyms = set(artifacts.get("known_acronyms", [])) | set(KNOWN_ACRONYM_SEED)

    def normalize_sentence(self, raw_words: List[str]) -> List[str]:
        return [self.normalize_token(raw_words, i) for i in range(len(raw_words))]

    def normalize_token(self, tokens: List[str], i: int) -> str:
        raw = str(tokens[i])
        low = raw.lower()

        protected = is_protected_token(raw, self.known_acronyms)
        exact_entry = self.exact.get(raw)

        if protected:
            if self._exact_is_strong(exact_entry, raw):
                return exact_entry["norm"]
            return raw

        candidates = [(raw, "copy", 0.0)]

        if self.config["use_exact_mfr"]:
            if self._exact_is_strong(exact_entry, raw):
                candidates.append((exact_entry["norm"], "exact_mfr", 6.0 + 3.0 * exact_entry["conf"]))

        if self.config["use_casefold_mfr"]:
            case_entry = self.casefold.get(low)
            if self._casefold_is_strong(case_entry, raw):
                candidates.append((case_entry["norm"], "casefold_mfr", 4.4 + 2.2 * case_entry["conf"]))

        if self.config["use_diacritics"]:
            for cand in self._diacritic_candidates(raw):
                candidates.append((cand, "diacritics", 4.0))

        if self.config["use_safe_abbrev"]:
            if low in SAFE_ABBREV:
                candidates.append((normalize_case_like(raw, SAFE_ABBREV[low]), "safe_abbrev", 3.7))

        if self.config["use_context_abbrev"]:
            for cand in self._context_candidates(tokens, i):
                candidates.append((cand, "context_abbrev", 3.0))

        if self.config["use_repeat"]:
            for cand in self._repeat_candidates(raw):
                candidates.append((cand, "repeat", 2.7))

        if self.config["use_capitalization"]:
            for cand in self._capitalization_candidates(raw, i):
                candidates.append((cand, "capitalization", 2.3))

        if self.config["use_split"]:
            if low in SPLIT_CANDIDATES:
                candidates.append((SPLIT_CANDIDATES[low], "split", 2.5))

        scored = []
        for cand, source, base in candidates:
            score = self._score_candidate(raw, cand, source, base, tokens, i)
            scored.append((score, cand, source))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        copy_score = self._copy_score(raw)
        best_score, best_cand, best_source = scored[0]

        if best_cand == raw:
            return raw

        if best_score - copy_score < self.config["decision_margin"]:
            return raw

        if " " in best_cand and best_source not in {"exact_mfr"}:
            if best_score - copy_score < self.config["decision_margin"] + 1.0:
                return raw

        return best_cand

    def _exact_is_strong(self, entry: Dict[str, Any], raw: str) -> bool:
        if not entry:
            return False
        if entry["norm"] == raw:
            return False
        return (
            entry["count"] >= self.config["min_count_exact"]
            and entry["conf"] >= self.config["min_conf_exact"]
        )

    def _casefold_is_strong(self, entry: Dict[str, Any], raw: str) -> bool:
        if not entry:
            return False
        if entry["norm"] == raw:
            return False
        if looks_like_named_entity(raw):
            return False
        return (
            entry["count"] >= self.config["min_count_casefold"]
            and entry["conf"] >= self.config["min_conf_casefold"]
        )

    def _copy_score(self, raw: str) -> float:
        score = 0.0

        if raw in self.norm_vocab:
            score += 1.55
        if raw.lower() in self.norm_vocab:
            score += 0.45
        if looks_like_named_entity(raw):
            score += 0.50
        if looks_like_acronym(raw, self.known_acronyms):
            score += 2.50
        if is_protected_token(raw, self.known_acronyms):
            score += 100.0

        return score

    def _score_candidate(
        self,
        raw: str,
        cand: str,
        source: str,
        base: float,
        tokens: List[str],
        i: int,
    ) -> float:
        if source == "copy":
            return self._copy_score(raw)

        score = base

        if cand in self.norm_vocab:
            score += 0.90 + min(1.20, math.log(self.norm_vocab[cand] + 1) / 4.0)
        elif cand.lower() in self.norm_vocab:
            score += 0.55

        if raw in self.norm_vocab and source not in {"exact_mfr", "casefold_mfr"}:
            score -= self.config["standard_raw_penalty"]

        if looks_like_named_entity(raw) and source not in {"exact_mfr", "casefold_mfr"}:
            score -= self.config["named_entity_penalty"]

        if " " in cand and source not in {"exact_mfr"}:
            score -= self.config["multiword_penalty"]

        score += self._context_bonus(raw, cand, source, tokens, i)

        return score

    def _context_bonus(self, raw: str, cand: str, source: str, tokens: List[str], i: int) -> float:
        low = raw.lower()
        prev_tok = tokens[i - 1].lower() if i > 0 else ""
        next_tok = tokens[i + 1].lower() if i + 1 < len(tokens) else ""

        bonus = 0.0

        if low == "x":
            if prev_tok.isdigit() or next_tok.isdigit():
                bonus -= 2.0
            else:
                bonus += 0.25

        if low == "6":
            if prev_tok.isdigit() or next_tok.isdigit():
                bonus -= 2.5
            elif next_tok in {"triste", "felice", "bello", "bella", "bravo", "brava", "grande"}:
                bonus += 1.0
            else:
                bonus += 0.15

        if low in {"qst", "qsto", "qsta", "qsti", "qste"}:
            expected = self._expected_questo_form(next_tok)
            if expected and cand == expected:
                bonus += 1.2
            elif expected and cand != expected:
                bonus -= 1.0

        if low == "tt":
            expected = self._expected_tutto_form(prev_tok, next_tok)
            if expected and cand == expected:
                bonus += 1.2
            elif expected and cand != expected:
                bonus -= 0.8

        if low == "c" and cand == "ci":
            if next_tok.endswith(("o", "i", "a", "e")):
                bonus += 0.20

        return bonus

    def _expected_questo_form(self, next_tok: str) -> str:
        if not next_tok:
            return ""

        if next_tok in {"ragazza", "cosa", "vita", "volta", "settimana"}:
            return "questa"
        if next_tok in {"ragazze", "cose", "volte"}:
            return "queste"
        if next_tok in {"ragazzo", "giorno", "mese", "anno", "momento"}:
            return "questo"
        if next_tok in {"giorni", "mesi", "anni", "momenti"}:
            return "questi"

        if next_tok.endswith("a"):
            return "questa"
        if next_tok.endswith("i"):
            return "questi"

        return ""

    def _expected_tutto_form(self, prev_tok: str, next_tok: str) -> str:
        if prev_tok == "a":
            return "tutti"

        if next_tok in {"i", "gli"}:
            return "tutti"
        if next_tok == "le":
            return "tutte"
        if next_tok in {"la", "una"}:
            return "tutta"
        if next_tok in {"il", "lo", "un"}:
            return "tutto"

        return ""

    def _context_candidates(self, tokens: List[str], i: int) -> List[str]:
        raw = str(tokens[i])
        low = raw.lower()

        if low not in CONTEXT_ABBREV:
            return []

        candidates = CONTEXT_ABBREV[low]

        if low == "qst":
            next_tok = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
            expected = self._expected_questo_form(next_tok)
            if expected:
                return [expected]
            return candidates

        if low == "tt":
            prev_tok = tokens[i - 1].lower() if i > 0 else ""
            next_tok = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
            expected = self._expected_tutto_form(prev_tok, next_tok)
            if expected:
                return [expected]
            return candidates

        return candidates

    def _diacritic_candidates(self, raw: str) -> List[str]:
        if raw in DIACRITIC_EXACT:
            return [DIACRITIC_EXACT[raw]]

        low = raw.lower()
        if low in DIACRITIC_LOWER:
            return [normalize_case_like(raw, DIACRITIC_LOWER[low])]

        if low in APOSTROPHE_PROTECTED:
            return []

        if len(raw) > 3 and raw[-1] in {"'", "`", "´", "’"}:
            stem = raw[:-1]
            last = stem[-1].lower()
            accent = {
                "a": "à",
                "e": "è",
                "i": "ì",
                "o": "ò",
                "u": "ù",
            }.get(last)

            if accent:
                return [stem[:-1] + accent]

        return []

    def _repeat_candidates(self, raw: str) -> List[str]:
        if not REPEAT_RE.search(raw):
            return []

        cand_one = REPEAT_RE.sub(r"\1", raw)
        cand_two = REPEAT_RE.sub(r"\1\1", raw)

        out = []
        for cand in [cand_one, cand_two]:
            if cand == raw:
                continue
            if cand in self.norm_vocab or cand.lower() in self.norm_vocab:
                out.append(cand)

        return list(dict.fromkeys(out))

    def _capitalization_candidates(self, raw: str, i: int) -> List[str]:
        out = []

        if len(raw) <= 1:
            return out

        if raw.isupper() and raw not in self.known_acronyms:
            lower = raw.lower()
            if lower in self.norm_vocab:
                out.append(lower)

        if i == 0 and raw.islower():
            titled = raw[:1].upper() + raw[1:]
            if titled in self.norm_vocab:
                out.append(titled)

        return out


def evaluate_metrics(raw_sents: List[List[str]], gold_sents: List[List[str]], pred_sents: List[List[str]]) -> Dict[str, float]:
    cor = 0
    changed = 0
    total = 0

    if len(gold_sents) != len(pred_sents):
        raise ValueError(f"gold sentences={len(gold_sents)}, pred sentences={len(pred_sents)}")

    for raw, gold, pred in zip(raw_sents, gold_sents, pred_sents):
        if len(gold) != len(pred):
            raise ValueError("A prediction sentence has different token length from gold.")

        for r, g, p in zip(raw, gold, pred):
            if r != g:
                changed += 1
            if g == p:
                cor += 1
            total += 1

    accuracy = cor / total if total else 0.0
    lai = (total - changed) / total if total else 0.0
    err = (accuracy - lai) / (1 - lai) if changed else 0.0

    return {
        "lai": lai,
        "accuracy": accuracy,
        "err": err,
        "total": total,
        "changed": changed,
    }