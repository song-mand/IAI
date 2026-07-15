from collections import Counter, defaultdict

KO_MASTER_MAP = {
    '조지고': '먹고', '홍들짝해서': '화들짝', '개꼴리네': '이끌린다',
    '틀딱들한테': '늙은이들한테', '아진짜': '정말', '맴찢일듯': '슬픈 것 같아',
    '애새끼': '아이', '개답없음': '매우답없음', '젖탱이': '젖가슴',
    '좋노': '좋다', '개씨발': '이런', '저딴집': '그런 집',
    '좆물받이': '물받이', '좆라도': '전라도', '그린놈': '그린친구',
    '넣은놈': '넣은친구', '미필새끼': '군대갔다오지 않은 사람',
    '회계사': '회계 전문가', '합격하고': '합격한 후', '3.5개월간': '3개월 반 동안',
    '수능': '대학수학능력시험', '공부했는데': '공부했지만', '개막장으로': '혼란으로',
    '개소리': '이상한 소리', '뚱돼지': '통통한 돼지', '많노': '많네',
    '년들': '나이 드는', '거르는거지': '선택하는거지', 'ㅈㄹ이냐': '행동이냐',
    '개쩌네': '너무좋다', '중궈': '중국인', '존나': '매우'
}

def build_ko_mfr(train_rows):
    """Compiles individual syllable maximum frequency tables."""
    counts = defaultdict(Counter)
    for row in train_rows:
        if row.get("lang", "ko") != "ko":
            continue
        for raw, norm in zip(row["raw"], row["norm"]):
            target = norm if norm is not None else raw
            counts[raw][target] += 1
            
    mfr_dict = {}
    for raw, counter in counts.items():
        mfr_dict[raw] = max(counter.items(), key=lambda x: (x[1], x[0] == raw))[0]
    return mfr_dict