#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

# This script trains temporary ES/IT holdout models and prints validation results.
# Use it for checking detector thresholds before final submission training.

python scripts/train_subfinal_models.py eval_es_it \
  --train_file ./data/train-00000-of-00001.parquet \
  --eval_dir ./eval_runs/es_it_hybrid \
  --test_size 0.2 \
  --seed 42 \
  --es_threshold 0.48 \
  --it_threshold 0.55 \
  --mfr_min_conf 0.65 \
  --es_it_epochs 5 \
  --es_it_lr 1e-5 \
  --unchanged_keep_prob 0.8 \
  --batch_size 16 \
  --detector_n_estimators 500 \
  --detector_max_depth 12 \
  --detector_min_samples_leaf 1 \
  --detector_min_samples_split 4 \
  --detector_max_features sqrt \
  --detector_class_weight custom \
  --detector_change_weight 3.5 \
  --verbose
