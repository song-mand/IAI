# Korean hard-coded normalization version

## Files

1. `ko_hardcode_rules.py`
   - Korean hard-coded rules + train-MFR logic.
   - Import this into your main `sub.py`.

2. `sub_ko_hardcoded_only.py`
   - Korean-only quick test script.
   - Produces `submission_ko_only.zip` for inspecting Korean predictions.

## How to use inside your full submission script

```python
from ko_hardcode_rules import build_ko_candidate_info, choose_ko_hardcoded

ko_train = train_split.filter(lambda x: x["lang"] == "ko")
ko_info = build_ko_candidate_info(ko_train)

pred_words = [
    choose_ko_hardcoded(
        raw_word=w,
        ko_info=ko_info,
        aggressive=False,
        trust_train=True,
        raw_keep_threshold=0.70,
        change_threshold=0.80,
    )
    for w in raw_words
]
```

## Tuning

### More aggressive Korean correction

```python
aggressive=True
```

This activates ambiguous mappings such as:

- 근데 -> 그런데
- 글고 -> 그리고
- 암튼 -> 아무튼
- 쫌 -> 좀

### More conservative train usage

Increase thresholds:

```python
raw_keep_threshold=0.60
change_threshold=0.90
```

### More aggressive train correction

```python
raw_keep_threshold=0.80
change_threshold=0.70
```
