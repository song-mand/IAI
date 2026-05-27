# ES / IT / KO ByT5 Lexical Normalization

## Files

1. `train_es_it_ko.py`
   - Trains separate ByT5-based models for Spanish (`es`), Italian (`it`), and Korean (`ko`).
   - Saves models to:
     - `./final_model/es_model`
     - `./final_model/it_model`
     - `./final_model/ko_model`

2. `sub_es_it_ko_rerank.py`
   - Creates `submission.zip`.
   - Uses ByT5 candidate generation + reranking for `es`, `it`, and `ko`.
   - Uses MFR fallback for languages without a trained model.
   - Korean additionally uses rule candidates and jamo-aware edit distance.

## Run

```bash
python train_es_it_ko.py
python sub_es_it_ko_rerank.py
```

## Main idea

The model may detect which token should be normalized, but may generate a slightly wrong normalized form.
So the inference code does not directly trust only the top-1 generated token.

For `es` and `it`:

1. Generate 3 candidates with ByT5.
2. Build a train-based candidate dictionary.
3. Use high-confidence MFR when the train evidence is stable.
4. If a model candidate matches a train candidate, use it.
5. If a model candidate is edit-distance close to a train candidate, correct to the train candidate.

For `ko`:

1. Generate 3 candidates with ByT5.
2. Add train candidates.
3. Add Korean rule candidates such as `조아 -> 좋아`, `마니 -> 많이`.
4. Use jamo-aware edit distance to handle Korean syllable-level similarity.
5. Rerank and choose the final prediction.

## Tuning points

In `sub_es_it_ko_rerank.py`, this threshold controls when MFR strongly overrides model predictions:

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

For Korean, if rule candidates are too aggressive, remove or comment out this part:

```python
if len(rule_candidates) == 1 and total <= 1:
    return rule_candidates[0]
```
