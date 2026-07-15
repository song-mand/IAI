from collections import Counter, defaultdict

# Global high-confidence macro mappings
JA_MACRO_MAP = {
    '大手': '大手企業', 
    'リモート': '在宅ワーク',
    'クッソョンハハァンデ': 'クッション ファン데이션',
    '際': '祭', 
    '決勝': '決勝戦', 
    'chara': 'Chara'
}

def build_ja_context(train_rows):
    """Compiles deterministic unigram, bigram, and trigram transition matrices."""
    unigram_counts = defaultdict(Counter)
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    
    for row in train_rows:
        if row.get("lang", "ja") != "ja":
            continue
        raw = row["raw"]
        norm = [n if n is not None else r for r, n in zip(row["raw"], row["norm"])]
        
        for i in range(len(raw)):
            r_curr = raw[i]
            n_curr = norm[i]
            
            unigram_counts[r_curr][n_curr] += 1
            if i < len(raw) - 1:
                bigram_counts[(r_curr, raw[i+1])][n_curr] += 1
            if i > 0 and i < len(raw) - 1:
                trigram_counts[(raw[i-1], r_curr, raw[i+1])][n_curr] += 1

    # Extract the maximum-frequency replacement for each context layer
    ja_context = {
        "unigram": {r: max(t.items(), key=lambda x: (x[1], x[0] == r))[0] for r, t in unigram_counts.items()},
        "bigram": {k: max(t.items(), key=lambda x: (x[1], x[0] == k[0]))[0] for k, t in bigram_counts.items()},
        "trigram": {k: max(t.items(), key=lambda x: (x[1], x[0] == k[1]))[0] for k, t in trigram_counts.items()}
    }
    return ja_context