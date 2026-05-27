# V4: Korean Classifier/Ranker + Conservative ES/IT

## Files

1. `train_es_it_v4.py`
   - Trains ByT5-based models only for Spanish (`es`) and Italian (`it`).
   - Korean is not trained as a generation model.

2. `sub_v4_es_it_ko.py`
   - Creates `submission.zip`.
   - Spanish / Italian use the conservative v3 logic:
     - ByT5 as cautious candidate generator
     - strong copy-protection
     - restricted norm-vocabulary lookup
   - Korean uses a non-hard-coded approach:
     - runtime copy/change classifier trained from Korean train split
     - train-derived candidate dictionary
     - jamo-distance norm-vocabulary retrieval
     - conservative candidate ranking

## Run

```bash
python train_es_it_v4.py
python sub_v4_es_it_ko.py
```

## Korean v4 idea

Previous Korean versions used either MFR or hand-written safe rules. V4 removes hard-coded slang mappings and instead trains a lightweight classifier at inference time from the provided train split.

Pipeline:

1. Build token-level Korean training examples.
2. Label each token as:
   - `KEEP` if `raw == norm`
   - `CHANGE` if `raw != norm`
3. Train char n-gram TF-IDF + Logistic Regression classifier.
4. During prediction:
   - If classifier says KEEP, keep raw.
   - If classifier says CHANGE, retrieve candidates from:
     - raw-specific train candidates
     - Korean norm vocabulary using jamo edit distance
   - Rank candidates conservatively.
   - If confidence is weak, keep raw.

## Requirements

The Korean classifier requires scikit-learn. If sklearn is not available, the code falls back to a very conservative candidate-only mode.

Recommended install if needed:

```bash
pip install scikit-learn
```

## Tuning points

In `choose_final_prediction_ko_classifier_ranker()`:

```python
if p_change < 0.60:
    return raw_word
```

More conservative:

```python
if p_change < 0.70:
    return raw_word
```

More aggressive:

```python
if p_change < 0.50:
    return raw_word
```

For unseen Korean raw tokens:

```python
if p_change < 0.75:
    return raw_word
```

Lowering this may increase recall but also increases overcorrection risk.
