#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

echo "=============================="
echo "0. Check required packages"
echo "=============================="
python - <<'PY'
import sklearn, joblib
print("scikit-learn/joblib OK")
PY

echo "=============================="
echo "1. Remove previous IT minimal MoNoise-style files"
echo "=============================="
rm -rf ./eval_splits_it_monoise
rm -rf ./models_it_monoise

mkdir -p ./models_it_monoise
mkdir -p ./reports

echo "=============================="
echo "2. Create IT holdout split"
echo "=============================="
python scripts/make_es_it_holdout.py \
  --source_train_file ./data/train-00000-of-00001.parquet \
  --langs it \
  --test_size 0.2 \
  --seed 42 \
  --out_dir ./eval_splits_it_monoise \
  --prefix it

echo "=============================="
echo "3. Train minimal MoNoise-style IT candidate ranker"
echo "=============================="
python monoise_applied/it_monoise_minimal.py train \
  --train_file ./eval_splits_it_monoise/it_train.parquet \
  --output ./models_it_monoise/it_monoise_minimal.joblib \
  --top_k_per_source 8 \
  --n_estimators 500 \
  --max_depth 16 \
  --min_samples_leaf 1 \
  --min_samples_split 2 \
  --max_features sqrt \
  --class_weight balanced_subsample \
  --seed 42

echo "=============================="
echo "4. Evaluate minimal MoNoise-style IT candidate ranker"
echo "=============================="
python monoise_applied/it_monoise_minimal.py eval \
  --train_file ./eval_splits_it_monoise/it_train.parquet \
  --valid_file ./eval_splits_it_monoise/it_valid.parquet \
  --model ./models_it_monoise/it_monoise_minimal.joblib \
  --margin 0.10 \
  --min_best_score 0.20 \
  --verbose | tee ./reports/it_monoise_minimal_eval.txt

echo "=============================="
echo "Done."
echo "Report:"
echo "  ./reports/it_monoise_minimal_eval.txt"
echo "=============================="
