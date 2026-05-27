# V2: ES / IT / KO Lexical Normalization

## Files

1. `train_es_it_v2.py`
   - Trains ByT5-based models for Spanish (`es`) and Italian (`it`) only.
   - Korean is intentionally not trained in this version.
   - Saved models:
     - `./final_model/es_model`
     - `./final_model/it_model`

2. `sub_v2_es_it_ko.py`
   - Creates `submission.zip`.
   - Spanish / Italian:
     - ByT5 candidate generation
     - raw-specific train candidates
     - MFR confidence
     - norm vocabulary candidates
     - accent-insensitive edit distance
     - overcorrection guard
   - Korean:
     - no ByT5 generation
     - conservative MFR
     - safe Korean rules only
     - unknown tokens are preserved

## Run

```bash
python train_es_it_v2.py
python sub_v2_es_it_ko.py
```

## Why Korean changed direction

The previous Korean ByT5 generation approach produced unstable correction results.
In v2, Korean is handled conservatively:

1. If the raw token was seen in train, use train-based MFR.
2. If it was not seen, apply only safe hand-written rules.
3. Otherwise, keep the original token.

This avoids unsupported Korean word generation.

## Why Spanish / Italian changed

Spanish and Italian still use ByT5, but the final output is not just the model top-1 answer.
The submission code searches for better candidates from:

1. model-generated candidates
2. raw-specific train candidates
3. global norm vocabulary candidates
4. MFR confidence

This helps when the model knows a token should change but generates a slightly wrong normalized form.

## Tuning points

### MFR strength for es/it

In `choose_final_prediction_es_it()`:

```python
if total >= 3 and best_ratio >= 0.90:
    return best
```

More MFR-heavy:

```python
if total >= 2 and best_ratio >= 0.85:
    return best
```

More model-heavy:

```python
if total >= 5 and best_ratio >= 0.95:
    return best
```

### Overcorrection guard

If es/it predictions are too conservative, loosen `overcorrection_guard()`.
If predictions change too many correct tokens, make it stricter.

### Korean safe rules

In `KOREAN_SAFE_RULES`, keep only highly reliable mappings.
Avoid ambiguous mappings such as:

- `ㄱㅅ -> 감사 / 고마워`
- `ㅈㅅ -> 죄송 / 미안`

unless train data supports them.
