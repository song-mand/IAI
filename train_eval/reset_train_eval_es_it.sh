#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

echo "=============================="
echo "1. Remove previous eval models/results"
echo "=============================="

rm -rf ./eval_splits
rm -rf ./final_model_eval
rm -rf ./models/byt5-es-eval
rm -rf ./models/byt5-it-eval

echo "Previous eval files removed."

echo "=============================="
echo "2. Create holdout split"
echo "=============================="

python scripts/make_es_it_holdout.py \
  --source_train_file ./data/train-00000-of-00001.parquet \
  --langs es it \
  --test_size 0.2 \
  --seed 42 \
  --out_dir ./eval_splits \
  --prefix es_it

echo "=============================="
echo "3. Train new ES/IT models"
echo "=============================="

python scripts/train_es_it_holdout.py \
  --train_file ./eval_splits/es_it_train.parquet \
  --langs es it \
  --output_model_dir ./final_model_eval \
  --seed 42 \
  --batch_size 16

echo "=============================="
echo "4. Evaluate new models"
echo "=============================="

python scripts/eval_es_it_holdout.py \
  --valid_file ./eval_splits/es_it_valid.parquet \
  --train_file ./eval_splits/es_it_train.parquet \
  --langs es it \
  --model_dir ./final_model_eval \
  --compare_mfr \
  --verbose

echo "=============================="
echo "Done."
echo "=============================="


nano reset_train_eval_es_it.sh               #<------execute

chmod +x reset_train_eval_es_it.sh
./reset_train_eval_es_it.sh