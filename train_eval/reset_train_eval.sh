#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

echo "=============================="
echo "1. Remove previous IT eval files"
echo "=============================="

rm -rf ./eval_splits_it
rm -rf ./final_model_eval_it
rm -rf ./models/byt5-it-eval

echo "Previous IT eval files removed."

echo "=============================="
echo "2. Create IT holdout split"
echo "=============================="

python scripts/make_es_it_holdout.py \
  --source_train_file ./data/train-00000-of-00001.parquet \
  --langs it \
  --test_size 0.2 \
  --seed 42 \
  --out_dir ./eval_splits_it \
  --prefix it

echo "=============================="
echo "3. Train IT model"
echo "=============================="

python scripts/train_es_it_holdout.py \
  --train_file ./eval_splits_it/it_train.parquet \
  --langs it \
  --output_model_dir ./final_model_eval_it \
  --seed 42 \
  --batch_size 16

echo "=============================="
echo "4. Evaluate IT model"
echo "=============================="

python scripts/eval_es_it_holdout.py \
  --valid_file ./eval_splits_it/it_valid.parquet \
  --train_file ./eval_splits_it/it_train.parquet \
  --langs it \
  --model_dir ./final_model_eval_it \
  --compare_mfr \
  --verbose

echo "=============================="
echo "IT train/eval done."
echo "=============================="