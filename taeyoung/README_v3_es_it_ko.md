# V3: Conservative ES / IT + Korean Rule MFR

## Files

1. `train_es_it_v3.py`
   - Trains ByT5-based models only for Spanish (`es`) and Italian (`it`).
   - Korean is not trained in this version.
   - Saved models:
     - `./final_model/es_model`
     - `./final_model/it_model`

2. `sub_v3_es_it_ko.py`
   - Creates `submission.zip`.
   - Spanish / Italian:
     - ByT5 is used only as a cautious candidate generator.
     - Strong copy-protection is added.
     - If train says the raw token is usually unchanged, raw is kept.
     - If raw was never seen in train, raw is kept unless model + restricted norm vocab strongly agree.
   - Korean:
     - No ByT5 generation.
     - Conservative train-based MFR + safe rules.
     - Unknown tokens are preserved.

## Run

```bash
python train_es_it_v3.py
python sub_v3_es_it_ko.py
```

## Why v3 changed

V2 caused severe overcorrection for Spanish and Italian.
The likely issue was that norm vocabulary reranking changed too many tokens that should have stayed unchanged.

V3 fixes that by adding strong copy-protection:

```python
if raw_ratio >= 0.70:
    return raw_word
```

and by making unseen raw tokens default to copy:

```python
if raw_word was not observed in train:
    keep raw unless model + restricted norm vocab strongly agree
```

## Korean direction

Korean improved when ByT5 generation was removed.
So v3 keeps the Korean conservative strategy:

1. If raw token is in train, use train-based MFR.
2. If unseen but in safe rules, apply the rule.
3. Otherwise keep raw.

## Tuning points

### Spanish / Italian copy protection

Current:

```python
if raw_ratio >= 0.70:
    return raw_word
```

More conservative:

```python
if raw_ratio >= 0.60:
    return raw_word
```

Less conservative:

```python
if raw_ratio >= 0.80:
    return raw_word
```

### Strong non-raw MFR

Current:

```python
if best != raw_word and total >= 3 and best_ratio >= 0.85:
    return best
```

More aggressive correction:

```python
if best != raw_word and total >= 2 and best_ratio >= 0.80:
    return best
```

More conservative correction:

```python
if best != raw_word and total >= 4 and best_ratio >= 0.90:
    return best
```
