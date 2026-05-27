# Korean contextual MFR reranker

Folder: `scripts/ko`

## 1. Train + validation check

```bash
python scripts/ko/train_ko_reranker.py \
  --train-path data/train-00000-of-00001.parquet \
  --valid-path data/validation-00000-of-00001.parquet \
  --model-dir artifacts/ko_reranker
```

## 2. Final model using train + validation

```bash
python scripts/ko/train_ko_reranker.py \
  --train-path data/train-00000-of-00001.parquet \
  --valid-path data/validation-00000-of-00001.parquet \
  --model-dir artifacts/ko_reranker \
  --fit-final-with-valid
```

## 3. Korean-only prediction

```bash
python scripts/ko/sub_ko_reranker.py \
  --test-path data/test-00000-of-00001.parquet \
  --model-dir artifacts/ko_reranker \
  --output-json submission_files/ko_predictions.json
```

## Required packages

```bash
pip install scikit-learn joblib tqdm pandas pyarrow
```

If you load from HuggingFace instead of local parquet files:

```bash
pip install datasets
```

## Bash runner

```bash
bash scripts/ko/run_ko_reranker.sh eval
bash scripts/ko/run_ko_reranker.sh final
bash scripts/ko/run_ko_reranker.sh predict
bash scripts/ko/run_ko_reranker.sh zip
```

If your parquet files are not in `data/`, set paths manually:

```bash
TRAIN_PATH=/path/to/train.parquet \
VALID_PATH=/path/to/validation.parquet \
TEST_PATH=/path/to/test.parquet \
bash scripts/ko/run_ko_reranker.sh final
```

## Full all-language submission

`run_ko_reranker.sh zip` is only for Korean-only experiments. For an actual submission,
use the full runner:

```bash
bash scripts/ko/run_ko_reranker_full.sh final
```

This creates `submission.zip` containing all test rows:

- `ko`: contextual reranker
- languages with an existing `final_model/{lang}_model`: seq2seq model
- all other languages: MFR dictionary

To reuse an already trained Korean reranker and only remake the full submission:

```bash
bash scripts/ko/run_ko_reranker_full.sh predict-all
```
