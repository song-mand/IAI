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
echo "1. Remove previous ES RF detector/hybrid files"
echo "=============================="
rm -rf ./eval_splits_es
rm -rf ./final_model_eval_es
rm -rf ./detectors/es_change_detector_rf.joblib
rm -rf ./models/byt5-es-eval

mkdir -p ./detectors
mkdir -p ./reports

echo "=============================="
echo "2. Create ES holdout split"
echo "=============================="
python scripts/make_es_it_holdout.py \
  --source_train_file ./data/train-00000-of-00001.parquet \
  --langs es \
  --test_size 0.2 \
  --seed 29 \
  --out_dir ./eval_splits_es \
  --prefix es

echo "=============================="
echo "3. Train ES ByT5 normalizer"
echo "=============================="
python scripts/train_es_it_holdout.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --langs es \
  --output_model_dir ./final_model_eval_es \
  --seed 29 \
  --batch_size 16

echo "=============================="
echo "4. Train ES RandomForest change detector"
echo "=============================="
python scripts/train_es_change_detector_rf.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --output ./detectors/es_change_detector_rf.joblib \
  --n_estimators 500 \
  --max_depth 12 \
  --min_samples_leaf 1 \
  --min_samples_split 4 \
  --max_features sqrt \
  --class_weight custom \
  --change_weight 3.5 \
  --random_state 42

#max_depth: low->simple, decrement of false positive, change recall // standard: 16
#min_sample_leaf: 1->detailed learning, many overfitting // 3 over->conservative, decrement of false positive, change recall
#min_samples_split: low-> aggresive, detailed// high-> conservative // standard: 4
#change_weight: low-> less change, also low change recall// high-> more change also more false positive
echo "=============================="
echo "5. Evaluate ES RandomForest change detector"
echo "=============================="
python scripts/eval_es_change_detector.py \
  --valid_file ./eval_splits_es/es_valid.parquet \
  --detector ./detectors/es_change_detector_rf.joblib \
  --threshold 0.5 \
  --show_errors | tee ./reports/es_detector_rf_eval.txt

echo "=============================="
echo "6. Evaluate ES RF detector + MFR + ByT5 hybrid"
echo "=============================="
python scripts/eval_es_hybrid_detector_byt5.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --valid_file ./eval_splits_es/es_valid.parquet \
  --detector ./detectors/es_change_detector_rf.joblib \
  --model_dir ./final_model_eval_es \
  --threshold 0.5 \
  --mode detector_mfr_byt5 \
  --mfr_min_conf 0.65 \
  --compare_direct_byt5 \
  --verbose | tee ./reports/es_hybrid_rf_eval.txt

echo "=============================="
echo "Done."
echo "Reports:"
echo "  ./reports/es_detector_rf_eval.txt"
echo "  ./reports/es_hybrid_rf_eval.txt"
echo "=============================="
