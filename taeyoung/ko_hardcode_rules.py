import re
from collections import Counter, defaultdict

"""
ko_hardcode_rules.py

Korean hard-coding focused normalization helper.

Usage:
    from ko_hardcode_rules import build_ko_candidate_info, choose_ko_hardcoded

    ko_info = build_ko_candidate_info(ko_train)
    pred_words = [choose_ko_hardcoded(w, ko_info) for w in raw_words]

Design:
- Korean is handled without ByT5 generation.
- Train-derived evidence is used first.
- Expanded hand-written Korean slang/abbreviation rules are used for unseen tokens.
- Unknown or risky tokens are kept unchanged.
"""

KO_DIRECT_RULES = {
    # date/time and common abbreviations
    "낼": "내일", "낼모레": "내일모레", "담주": "다음주", "담달": "다음달",
    "담번": "다음번", "담에": "다음에", "낼봐": "내일 봐",
    "짐": "지금", "지금머해": "지금 뭐해",

    # adverbs / casual spellings
    "걍": "그냥", "넘": "너무", "느무": "너무", "넘나": "너무나",
    "진짜루": "진짜로", "진쨔": "진짜", "진짜아": "진짜",

    # questions / spoken forms
    "머": "뭐", "머해": "뭐해", "모해": "뭐해", "머함": "뭐함", "모함": "뭐함",
    "뭐하구": "뭐하고", "모하구": "뭐하고",
    "어케": "어떻게", "어캐": "어떻게", "어케해": "어떻게 해",
    "어카지": "어떡하지", "어카냐": "어떡하냐",
    "왤케": "왜 이렇게", "왜케": "왜 이렇게", "일케": "이렇게", "글케": "그렇게",

    # phonetic variants
    "마니": "많이", "마니마니": "많이 많이",
    "조아": "좋아", "조앙": "좋아", "조아해": "좋아해", "조아용": "좋아요",
    "시러": "싫어", "시름": "싫음", "싫엉": "싫어",
    "마자": "맞아", "마쟈": "맞아", "맞앙": "맞아",
    "아냐": "아니야", "아뉘": "아니", "아니얌": "아니야",
    "안대": "안돼", "안되": "안돼", "안돼여": "안돼요", "돼게": "되게",

    # greetings / reactions
    "안뇽": "안녕", "안뇽하세여": "안녕하세요", "안냐세여": "안녕하세요",
    "안녕하세용": "안녕하세요", "방가": "반가워", "방가방가": "반가워",
    "하이루": "안녕", "바이루": "잘가", "굿밤": "좋은 밤", "굿모닝": "좋은 아침",

    # chat abbreviations
    "ㅇㅇ": "응", "ㅇㅋ": "오케이", "오키": "오케이", "오케": "오케이",
    "ㄴㄴ": "아니", "노노": "아니", "ㄱㄱ": "고고",
    "ㄱㅊ": "괜찮아", "괜춘": "괜찮아", "괜차나": "괜찮아",
    "ㄹㅇ": "레알", "ㅇㅈ": "인정", "ㅁㅈ": "맞아", "ㅅㄱ": "수고", "ㅊㅋ": "축하",

    # thanks / apology: somewhat aggressive, but useful in hard-code mode
    "ㅈㅅ": "죄송", "ㅈㅅㅈㅅ": "죄송 죄송", "죄송여": "죄송해요",
    "미안해용": "미안해요", "ㄱㅅ": "감사", "ㄳ": "감사", "감사여": "감사해요",
    "고마워용": "고마워요",

    # intensifier / slang
    "개좋아": "정말 좋아", "개웃겨": "정말 웃겨", "개웃김": "정말 웃김",
    "짱좋아": "정말 좋아", "존맛": "정말 맛있다", "존맛탱": "정말 맛있다",
    "노잼": "재미없다", "꿀잼": "재미있다",

    # spacing-like common variants
    "집가는중": "집 가는 중", "가는중": "가는 중", "하는중": "하는 중",
}

KO_AMBIGUOUS_RULES = {
    "근데": "그런데", "근대": "그런데", "글고": "그리고", "암튼": "아무튼",
    "쫌": "좀", "좀만": "조금만", "먼가": "뭔가", "몬가": "뭔가",
    "먼데": "뭔데", "머야": "뭐야", "모야": "뭐야", "머임": "뭐임", "모임": "뭐임",
}

ENDING_RULES = [
    (re.compile(r"(.+)해용$"), r"\1해요"),
    (re.compile(r"(.+)에용$"), r"\1예요"),
    (re.compile(r"(.+)이에용$"), r"\1이에요"),
    (re.compile(r"(.+)네용$"), r"\1네요"),
    (re.compile(r"(.+)거든용$"), r"\1거든요"),
    (re.compile(r"(.+)잖아용$"), r"\1잖아요"),
]


def is_keep_token(token):
    if token is None or token == "":
        return True
    if token.startswith("@") or token.startswith("#") or token.startswith("http"):
        return True
    if re.fullmatch(r"[0-9]+([:.,/-][0-9]+)*", token):
        return True
    if re.fullmatch(r"[\W_]+", token, flags=re.UNICODE):
        return True
    return False


def collapse_repeated_emote(token):
    if not re.fullmatch(r"[ㅋㅎㅠㅜ]+", token):
        return None
    if len(set(token)) != 1:
        return None
    if len(token) >= 4:
        return token[0] * 2
    return None


def reduce_repeated_syllables(token):
    m = re.match(r"^(.+?)(.)\2{2,}$", token)
    if not m:
        return None
    stem, repeated = m.group(1), m.group(2)
    if repeated in "ㅋㅎㅠㅜ":
        return None
    return stem + repeated


def apply_ending_rules(token):
    for pattern, repl in ENDING_RULES:
        if pattern.fullmatch(token):
            return pattern.sub(repl, token)
    return None


def build_ko_candidate_info(ko_train_data):
    counts = defaultdict(Counter)
    for row in ko_train_data:
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1

    info = {}
    for raw, target_counter in counts.items():
        total = sum(target_counter.values())
        best, best_count = target_counter.most_common(1)[0]
        raw_count = target_counter.get(raw, 0)
        info[raw] = {
            "total": total,
            "best": best,
            "best_count": best_count,
            "best_ratio": best_count / total if total else 0.0,
            "raw_count": raw_count,
            "raw_ratio": raw_count / total if total else 0.0,
            "candidates": target_counter,
        }
    return info


def choose_ko_hardcoded(raw_word, ko_info, aggressive=False, trust_train=True,
                         raw_keep_threshold=0.70, change_threshold=0.80):
    if is_keep_token(raw_word):
        return raw_word

    if trust_train and raw_word in ko_info:
        item = ko_info[raw_word]
        best = item["best"]
        total = item["total"]
        best_ratio = item["best_ratio"]
        raw_ratio = item["raw_ratio"]

        if raw_ratio >= raw_keep_threshold:
            return raw_word
        if best != raw_word and total >= 2 and best_ratio >= change_threshold:
            return best
        if raw_word in KO_DIRECT_RULES and total <= 2:
            return KO_DIRECT_RULES[raw_word]
        return raw_word

    if raw_word in KO_DIRECT_RULES:
        return KO_DIRECT_RULES[raw_word]
    if aggressive and raw_word in KO_AMBIGUOUS_RULES:
        return KO_AMBIGUOUS_RULES[raw_word]

    collapsed = collapse_repeated_emote(raw_word)
    if collapsed is not None:
        return collapsed

    reduced = reduce_repeated_syllables(raw_word)
    if reduced is not None:
        return reduced

    ended = apply_ending_rules(raw_word)
    if ended is not None:
        return ended

    return raw_word


def predict_ko_sentence(raw_words, ko_info, aggressive=False):
    return [choose_ko_hardcoded(w, ko_info, aggressive=aggressive) for w in raw_words]
