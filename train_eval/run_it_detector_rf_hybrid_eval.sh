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
echo "1. Remove previous IT RF detector/hybrid files"
echo "=============================="
rm -rf ./eval_splits_it
rm -rf ./final_model_eval_it
rm -rf ./detectors/it_change_detector_rf.joblib
rm -rf ./models/byt5-it-eval

mkdir -p ./detectors
mkdir -p ./reports

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
echo "3. Train IT ByT5 normalizer"
echo "=============================="
python scripts/train_es_it_holdout.py \
  --train_file ./eval_splits_it/it_train.parquet \
  --langs it \
  --output_model_dir ./final_model_eval_it \
  --seed 42 \
  --batch_size 16

echo "=============================="
echo "4. Train IT RandomForest change detector"
echo "=============================="
python scripts/train_it_change_detector_rf.py \
  --train_file ./eval_splits_it/it_train.parquet \
  --output ./detectors/it_change_detector_rf.joblib \
  --n_estimators 500 \
  --max_depth 12 \
  --min_samples_leaf 1 \
  --min_samples_split 4 \
  --max_features sqrt \
  --class_weight custom \
  --change_weight 3.5 \
  --random_state 42

echo "=============================="
echo "5. Evaluate IT RandomForest change detector"
echo "=============================="
python scripts/eval_it_change_detector.py \
  --valid_file ./eval_splits_it/it_valid.parquet \
  --detector ./detectors/it_change_detector_rf.joblib \
  --threshold 0.5 \
  --show_errors | tee ./reports/it_detector_rf_eval.txt

echo "=============================="
echo "6. Evaluate IT RF detector + MFR + ByT5 hybrid"
echo "=============================="
python scripts/eval_it_hybrid_detector_byt5.py \
  --train_file ./eval_splits_it/it_train.parquet \
  --valid_file ./eval_splits_it/it_valid.parquet \
  --detector ./detectors/it_change_detector_rf.joblib \
  --model_dir ./final_model_eval_it \
  --threshold 0.5 \
  --byt5_threshold 0.65 \
  --mode detector_mfr_byt5 \
  --mfr_min_conf 0.65 \
  --compare_direct_byt5 \
  --verbose | tee ./reports/it_hybrid_rf_eval.txt

echo "=============================="
echo "Done."
echo "Reports:"
echo "  ./reports/it_detector_rf_eval.txt"
echo "  ./reports/it_hybrid_rf_eval.txt"
echo "=============================="
