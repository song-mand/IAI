#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

echo "=============================="
echo "1. Remove previous ES action-aware eval files"
echo "=============================="

rm -rf ./eval_splits_es
rm -rf ./final_model_eval_es_action
rm -rf ./models/byt5-es-action-eval

echo "Previous ES action-aware files removed."

echo "=============================="
echo "2. Create ES holdout split"
echo "=============================="

python scripts/make_es_it_holdout.py \
  --source_train_file ./data/train-00000-of-00001.parquet \
  --langs es \
  --test_size 0.2 \
  --seed 42 \
  --out_dir ./eval_splits_es \
  --prefix es

echo "=============================="
echo "3. Train ES action-aware model"
echo "=============================="

python scripts/train_es_action_holdout.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --output_model_dir ./final_model_eval_es_action \
  --seed 42 \
  --batch_size 16 \
  --epochs 2 \
  --learning_rate 2e-5 \
  --copy_keep_prob 0.7 \
  --changed_repeat 3 \
  --lora_target all-linear   #qv / all-linear

echo "=============================="
echo "4. Evaluate ES action-aware model"
echo "=============================="

python scripts/eval_es_action_holdout.py \
  --valid_file ./eval_splits_es/es_valid.parquet \
  --train_file ./eval_splits_es/es_train.parquet \
  --model_dir ./final_model_eval_es_action \
  --compare_mfr \
  --verbose

echo "=============================="
echo "ES action-aware train/eval done."
echo "=============================="
