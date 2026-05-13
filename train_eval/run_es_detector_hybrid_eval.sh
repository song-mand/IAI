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
echo "1. Remove previous ES detector/hybrid files"
echo "=============================="
rm -rf ./eval_splits_es
rm -rf ./final_model_eval_es
rm -rf ./detectors/es_change_detector.joblib
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
  --seed 5 \
  --out_dir ./eval_splits_es \
  --prefix es

echo "=============================="
echo "3. Train ES ByT5 normalizer"
echo "=============================="
python scripts/train_es_it_holdout.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --langs es \
  --output_model_dir ./final_model_eval_es \
  --seed 42 \
  --batch_size 16

echo "=============================="
echo "4. Train ES change detector"
echo "=============================="
python scripts/train_es_change_detector.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --output ./detectors/es_change_detector.joblib

echo "=============================="
echo "5. Evaluate ES change detector"
echo "=============================="
python scripts/eval_es_change_detector.py \
  --valid_file ./eval_splits_es/es_valid.parquet \
  --detector ./detectors/es_change_detector.joblib \
  --threshold 0.5 \
  --show_errors | tee ./reports/es_detector_eval.txt

echo "=============================="
echo "6. Evaluate ES detector + MFR + ByT5 hybrid"
echo "=============================="
python scripts/eval_es_hybrid_detector_byt5.py \
  --train_file ./eval_splits_es/es_train.parquet \
  --valid_file ./eval_splits_es/es_valid.parquet \
  --detector ./detectors/es_change_detector.joblib \
  --model_dir ./final_model_eval_es \
  --threshold 0.5 \
  --mode detector_mfr_byt5 \
  --mfr_min_conf 0.60 \
  --compare_direct_byt5 \
  --verbose | tee ./reports/es_hybrid_eval.txt

# threshold: must be same as threshold of evaluation of detector.

# mode: three mode avaliable:
#       1. detector_mfr_byt5: detector determine change or copy, if change, mfr or byt5 is used.
#                                                                -->mfr: if confidence is sufficient 
#       2. detector_mfr_only: no byt5
#       3. detector_byt5: no mfr 

#mfr_min_conf: min confidence of mfr candidates. low confidence->more mfr 

echo "=============================="
echo "Done."
echo "Reports:"
echo "  ./reports/es_detector_eval.txt"
echo "  ./reports/es_hybrid_eval.txt"
echo "=============================="
