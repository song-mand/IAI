#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

rm -rf ./eval_splits_es
rm -rf ./final_model_eval_es
rm -rf ./models/byt5-es-eval

python scripts/make_es_it_holdout.py \
  --source_train_file ./data/train-00000-of-00001.parquet \
  --langs es \
  --test_size 0.2 \
  --seed 42 \
  --out_dir ./eval_splits_es \
  --prefix es

python scripts/train_es_it_holdout.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --langs es \
  --output_model_dir ./final_model_eval_es \
  --seed 42 \
  --batch_size 16

python scripts/eval_es_it_holdout.py \
  --valid_file ./eval_splits_es/es_valid.parquet \
  --train_file ./eval_splits_es/es_train.parquet \
  --langs es \
  --model_dir ./final_model_eval_es \
  --compare_mfr \
  --verbose



./reset_train_eval_es.sh
