import math
import pickle
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)
MENTION_RE = re.compile(r"^@\w+")
HASHTAG_RE = re.compile(r"^#\S+")
PUNCT_ONLY_RE = re.compile(r"^[^\wÀ-ÖØ-öø-ÿ]+$")
REPEAT_RE = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ])\1{2,}")

SAFE_ABBREV = {
    "cmq": "comunque",
    "nn": "non",
    "nnt": "niente",
    "ke": "che",
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
    "grz": "grazie",
    "qnd": "quando",
    "cn": "con",
    "dv": "dove",
    "x": "per",
}

DIACRITIC_MAP = {
    "e'": "è",
    "e`": "è",
    "e´": "è",
    "E'": "È",
    "E`": "È",
    "E´": "È",
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


def safe_norm(raw: str, norm: Any) -> str:
    return raw if norm is None else str(norm)


def token_shape(tok: str) -> str:
    if not tok:
        return "EMPTY"
    if URL_RE.match(tok):
        return "URL"
    if MENTION_RE.match(tok):
        return "MENTION"
    if HASHTAG_RE.match(tok):
        return "HASHTAG"
    if PUNCT_ONLY_RE.match(tok):
        return "PUNCT"
    if tok.isupper():
        return "ALLCAPS"
    if tok.islower():
        return "LOWER"
    if tok[:1].isupper():
        return "TITLE"
    if any(ch.isdigit() for ch in tok):
        return "HASDIGIT"
    return "MIXED"


def is_protected(tok: str) -> bool:
    return (
        URL_RE.match(tok) is not None
        or MENTION_RE.match(tok) is not None
        or HASHTAG_RE.match(tok) is not None
        or PUNCT_ONLY_RE.match(tok) is not None
    )


def edit_distance(a: str, b: str, max_cutoff: int = 4) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_cutoff:
        return max_cutoff + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = i

        for j, cb in enumerate(b, start=1):
            ins = cur[-1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            v = min(ins, dele, sub)
            cur.append(v)
            row_min = min(row_min, v)

        if row_min > max_cutoff:
            return max_cutoff + 1

        prev = cur

    return prev[-1]


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0

    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)

    return ent


def build_mfr_counts(rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    counts = defaultdict(Counter)

    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            raw = str(raw)
            norm = safe_norm(raw, norm)
            counts[raw][norm] += 1

    return counts


def build_vocab(rows: List[Dict[str, Any]]) -> Tuple[Counter, Counter]:
    raw_vocab = Counter()
    norm_vocab = Counter()

    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            raw = str(raw)
            norm = safe_norm(raw, norm)
            raw_vocab[raw] += 1
            norm_vocab[norm] += 1

    return raw_vocab, norm_vocab


def choose_mfr(counter: Counter, raw: str) -> str:
    if not counter:
        return raw

    max_count = max(counter.values())
    bests = [k for k, v in counter.items() if v == max_count]

    if raw in bests:
        return raw

    return sorted(bests)[0]


def build_mfr_dict(mfr_counts: Dict[str, Counter]) -> Dict[str, str]:
    return {
        raw: choose_mfr(counter, raw)
        for raw, counter in mfr_counts.items()
    }


class ITCandidateGenerator:
    def __init__(
        self,
        mfr_counts: Dict[str, Counter],
        raw_vocab: Counter,
        norm_vocab: Counter,
        use_casefold: bool = False,
        use_diacritics: bool = False,
        use_repeat: bool = False,
        use_abbrev: bool = False,
    ):
        self.mfr_counts = mfr_counts
        self.mfr_dict = build_mfr_dict(mfr_counts)
        self.raw_vocab = raw_vocab
        self.norm_vocab = norm_vocab

        self.use_casefold = use_casefold
        self.use_diacritics = use_diacritics
        self.use_repeat = use_repeat
        self.use_abbrev = use_abbrev

        self.casefold_counts = defaultdict(Counter)
        for raw, counter in mfr_counts.items():
            for norm, c in counter.items():
                self.casefold_counts[raw.lower()][norm] += c

        self.casefold_mfr = build_mfr_dict(self.casefold_counts)

    def candidates(self, raw: str, gold: str = None) -> List[str]:
        raw = str(raw)
        cands = [raw]

        if is_protected(raw):
            if gold is not None and gold not in cands:
                cands.append(gold)
            return cands

        if raw in self.mfr_dict:
            cands.append(self.mfr_dict[raw])

        low = raw.lower()

        if self.use_casefold and low in self.casefold_mfr:
            cands.append(self.casefold_mfr[low])

        if self.use_diacritics:
            if raw in DIACRITIC_MAP:
                cands.append(DIACRITIC_MAP[raw])
            elif low in DIACRITIC_MAP:
                mapped = DIACRITIC_MAP[low]
                if raw[:1].isupper():
                    mapped = mapped[:1].upper() + mapped[1:]
                cands.append(mapped)

        if self.use_repeat:
            cands.extend(self._repeat_candidates(raw))

        if self.use_abbrev and low in SAFE_ABBREV:
            mapped = SAFE_ABBREV[low]
            if raw[:1].isupper():
                mapped = mapped[:1].upper() + mapped[1:]
            cands.append(mapped)

        if gold is not None and gold not in cands:
            cands.append(gold)

        # 순서 유지 중복 제거
        return list(dict.fromkeys(cands))

    def _repeat_candidates(self, raw: str) -> List[str]:
        if REPEAT_RE.search(raw) is None:
            return []

        c1 = REPEAT_RE.sub(r"\1", raw)
        c2 = REPEAT_RE.sub(r"\1\1", raw)

        out = []
        for c in [c1, c2]:
            if c != raw and (c in self.norm_vocab or c.lower() in self.norm_vocab):
                out.append(c)

        return out

    def mfr_info(self, raw: str, cand: str) -> Dict[str, float]:
        counter = self.mfr_counts.get(raw, Counter())
        total = sum(counter.values())
        cnt = counter.get(cand, 0)
        conf = cnt / total if total else 0.0
        ent = entropy(counter)

        top = self.mfr_dict.get(raw, raw)

        return {
            "mfr_count": cnt,
            "mfr_total": total,
            "mfr_conf": conf,
            "mfr_entropy": ent,
            "is_mfr_top": int(cand == top),
        }


class ITCandidateRanker:
    def __init__(
        self,
        generator: ITCandidateGenerator,
        threshold: float = 0.50,
        C: float = 1.0,
    ):
        self.generator = generator
        self.threshold = threshold
        self.C = C
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(
            max_iter=1000,
            C=C,
            class_weight="balanced",
            solver="liblinear",
        )

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        X_dicts = []
        y = []

        for row in rows:
            raw_words = [str(x) for x in row["raw"]]
            norm_words = [
                safe_norm(str(r), n)
                for r, n in zip(row["raw"], row["norm"])
            ]

            for i, (raw, gold) in enumerate(zip(raw_words, norm_words)):
                cands = self.generator.candidates(raw, gold=gold)

                for cand in cands:
                    X_dicts.append(self.features(raw_words, i, cand))
                    y.append(int(cand == gold))

        X = self.vectorizer.fit_transform(X_dicts)
        self.model.fit(X, y)

    def predict_sentence(self, raw_words: List[str]) -> List[str]:
        raw_words = [str(x) for x in raw_words]
        pred = []

        for i, raw in enumerate(raw_words):
            cands = self.generator.candidates(raw, gold=None)

            if len(cands) == 1:
                pred.append(cands[0])
                continue

            X_dicts = [self.features(raw_words, i, cand) for cand in cands]
            X = self.vectorizer.transform(X_dicts)

            probs = self.model.predict_proba(X)[:, 1]

            best_idx = int(probs.argmax())
            best_cand = cands[best_idx]
            best_prob = float(probs[best_idx])

            if best_cand != raw and best_prob < self.threshold:
                pred.append(raw)
            else:
                pred.append(best_cand)

        return pred

    def predict_rows(self, rows: List[Dict[str, Any]]) -> List[List[str]]:
        return [self.predict_sentence(row["raw"]) for row in rows]

    def features(self, sent: List[str], i: int, cand: str) -> Dict[str, Any]:
        raw = sent[i]
        low = raw.lower()
        cand_low = cand.lower()

        prev_tok = sent[i - 1] if i > 0 else "<BOS>"
        next_tok = sent[i + 1] if i + 1 < len(sent) else "<EOS>"

        mfr = self.generator.mfr_info(raw, cand)

        all_alpha = [t for t in sent if any(ch.isalpha() for ch in t)]
        allcaps_count = sum(1 for t in all_alpha if t.isupper() and len(t) > 1)
        allcaps_ratio = allcaps_count / len(all_alpha) if all_alpha else 0.0

        ed = edit_distance(raw, cand)

        cap_only = int(raw.lower() == cand.lower() and raw != cand)
        diacritic_like = int(
            raw.replace("'", "").replace("`", "").replace("´", "").lower()
            != raw.lower()
            or any(ch in raw + cand for ch in "àèéìòùÀÈÉÌÒÙ")
        )

        feats = {
            # raw/candidate identity
            "raw_lower": low,
            "cand_lower": cand_low,
            "pair": f"{low}->{cand_low}",

            # local context
            "prev_lower": prev_tok.lower(),
            "next_lower": next_tok.lower(),
            "prev_shape": token_shape(prev_tok),
            "next_shape": token_shape(next_tok),
            "prev_cand_bigram": f"{prev_tok.lower()}_{cand_low}",
            "cand_next_bigram": f"{cand_low}_{next_tok.lower()}",

            # shapes
            "raw_shape": token_shape(raw),
            "cand_shape": token_shape(cand),
            "is_copy": int(raw == cand),
            "is_changed": int(raw != cand),
            "is_cap_only": cap_only,
            "is_diacritic_like": diacritic_like,
            "is_protected": int(is_protected(raw)),

            # sentence position
            "position_bucket": self._position_bucket(i, len(sent)),
            "is_first": int(i == 0),
            "is_last": int(i == len(sent) - 1),

            # sentence typography
            "sent_allcaps_ratio_bin": self._ratio_bin(allcaps_ratio),

            # MFR statistics
            "mfr_conf": mfr["mfr_conf"],
            "mfr_count_log": math.log(mfr["mfr_count"] + 1),
            "mfr_total_log": math.log(mfr["mfr_total"] + 1),
            "mfr_entropy": mfr["mfr_entropy"],
            "is_mfr_top": mfr["is_mfr_top"],

            # vocabulary
            "raw_seen_log": math.log(self.generator.raw_vocab.get(raw, 0) + 1),
            "cand_norm_seen_log": math.log(self.generator.norm_vocab.get(cand, 0) + 1),
            "cand_norm_lower_seen": int(cand_low in {k.lower() for k in self.generator.norm_vocab.keys()}),

            # edit information
            "edit_distance": ed,
            "length_delta": len(cand) - len(raw),
            "raw_len_bin": self._len_bin(len(raw)),
            "cand_len_bin": self._len_bin(len(cand)),
        }

        return feats

    def _position_bucket(self, i: int, n: int) -> str:
        if n <= 1:
            return "only"
        ratio = i / (n - 1)
        if ratio < 0.2:
            return "start"
        if ratio < 0.8:
            return "middle"
        return "end"

    def _ratio_bin(self, x: float) -> str:
        if x == 0:
            return "0"
        if x < 0.25:
            return "low"
        if x < 0.60:
            return "mid"
        return "high"

    def _len_bin(self, n: int) -> str:
        if n <= 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 6:
            return "4-6"
        if n <= 10:
            return "7-10"
        return "11+"

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ITCandidateRanker":
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate(raw_sents: List[List[str]], gold_sents: List[List[str]], pred_sents: List[List[str]]) -> Dict[str, float]:
    total = 0
    correct = 0
    changed_gold = 0

    for raw, gold, pred in zip(raw_sents, gold_sents, pred_sents):
        if len(raw) != len(gold) or len(gold) != len(pred):
            raise ValueError("Length mismatch during evaluation.")

        for r, g, p in zip(raw, gold, pred):
            if r != g:
                changed_gold += 1
            if g == p:
                correct += 1
            total += 1

    lai = (total - changed_gold) / total if total else 0.0
    acc = correct / total if total else 0.0
    err = (acc - lai) / (1 - lai) if changed_gold else 0.0

    return {
        "lai": lai,
        "accuracy": acc,
        "err": err,
        "total": total,
        "changed_gold": changed_gold,
    }


def make_ranker(
    train_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    threshold: float,
) -> ITCandidateRanker:
    mfr_counts = build_mfr_counts(train_rows)
    raw_vocab, norm_vocab = build_vocab(train_rows)

    generator = ITCandidateGenerator(
        mfr_counts=mfr_counts,
        raw_vocab=raw_vocab,
        norm_vocab=norm_vocab,
        use_casefold=config.get("use_casefold", False),
        use_diacritics=config.get("use_diacritics", False),
        use_repeat=config.get("use_repeat", False),
        use_abbrev=config.get("use_abbrev", False),
    )

    ranker = ITCandidateRanker(
        generator=generator,
        threshold=threshold,
        C=config.get("C", 1.0),
    )

    ranker.fit(train_rows)
    return ranker