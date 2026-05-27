#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

# ============================================================
# run_es_oof_experiment.sh
#
# Purpose:
#   1. Optionally train ES OOF ranker system.
#   2. Create submission.zip with:
#      - ES: OOF ranker + ByT5 candidate-specific margin
#      - Other languages: MFR
#
# Recommended now:
#   TRAIN_BEFORE_SUBMISSION=0
#   because only the submission code was changed.
# ============================================================

# ------------------------------
# Paths
# ------------------------------
TRAIN_FILE="./data/train-00000-of-00001.parquet"

TRAIN_SCRIPT="scripts/es/train_es_oof_ranker.py"
SUBMISSION_SCRIPT="scripts/es/sub_es_ranker_oof_all_mfr.py"

ES_MODEL_DIR="./final_model/es_model"
ES_DETECTOR="./detectors/es_change_detector_rf.joblib"
ES_RANKER="./detectors/es_candidate_ranker_oof_rf.joblib"
ES_RESOURCES="./detectors/es_resources_oof.joblib"

OUTPUT_JSON="./submission_files/predictions.json"
OUTPUT_ZIP="submission.zip"

# ------------------------------
# Run options
# ------------------------------

# 1 = train detector/ranker/resources before submission
# 0 = use existing detector/ranker/resources and only make submission
TRAIN_BEFORE_SUBMISSION=1

# ------------------------------
# Detector hyperparameters
# ------------------------------
DETECTOR_THRESHOLD=0.43
DETECTOR_N_ESTIMATORS=600
DETECTOR_MAX_DEPTH=12
DETECTOR_MIN_SAMPLES_LEAF=1
DETECTOR_MIN_SAMPLES_SPLIT=4
DETECTOR_MAX_FEATURES="sqrt"
DETECTOR_CLASS_WEIGHT="custom"
DETECTOR_CHANGE_WEIGHT=3.5

# ------------------------------
# OOF ranker hyperparameters
# ------------------------------
RANKER_OOF_FOLDS=5
RANKER_UNCHANGED_KEEP_PROB=0.20

RANKER_N_ESTIMATORS=800
RANKER_MAX_DEPTH=14
RANKER_MIN_SAMPLES_LEAF=1
RANKER_MIN_SAMPLES_SPLIT=4
RANKER_MAX_FEATURES="sqrt"
RANKER_CLASS_WEIGHT="none"
RANKER_POSITIVE_WEIGHT=1.0
RANKER_THRESHOLD=0.25

# Normal candidate margin.
# Lower = more aggressive non-copy selection.
# Higher = safer, more copy.
CANDIDATE_MARGIN=0.03

# ------------------------------
# Ranker sample weights
# ------------------------------
CHANGED_POSITIVE_WEIGHT=4.0
CHANGED_COPY_WRONG_WEIGHT=3.0
CHANGED_WRONG_WEIGHT=1.5
UNCHANGED_COPY_POSITIVE_WEIGHT=0.3
OVERCHANGE_NEGATIVE_WEIGHT=2.0

# ------------------------------
# Submission hyperparameters
# ------------------------------

# Detector probability gate for asking ByT5 after ranker chose copy.
# Use 0.43 to reopen ByT5 fallback similarly to the earlier high-ERR setting.
BYT5_COPY_DETECTOR_THRESHOLD=0.43

# Separate margin for accepting ByT5 candidate.
# Lower = ByT5 candidate is accepted more easily.
# Start with -0.05. If over-change is too high, try -0.03 or 0.00.
BYT5_CANDIDATE_MARGIN=-0.15

NUM_BEAMS=1
MAX_NEW_TOKENS=12
BYT5_GENERATE_BATCH_SIZE=64

echo "=============================="
echo "0. Check files and packages"
echo "=============================="

python - <<'PY'
import torch, sklearn, joblib, transformers, datasets
print("packages OK")
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

if [ ! -f "$TRAIN_FILE" ]; then
  echo "ERROR: TRAIN_FILE not found: $TRAIN_FILE"
  exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "ERROR: TRAIN_SCRIPT not found: $TRAIN_SCRIPT"
  exit 1
fi

if [ ! -f "$SUBMISSION_SCRIPT" ]; then
  echo "ERROR: SUBMISSION_SCRIPT not found: $SUBMISSION_SCRIPT"
  exit 1
fi

if [ ! -d "$ES_MODEL_DIR" ]; then
  echo "ERROR: ES ByT5 model directory not found: $ES_MODEL_DIR"
  echo "Train ES ByT5 first or copy it to ./final_model/es_model."
  exit 1
fi

mkdir -p ./detectors
mkdir -p ./submission_files

echo "=============================="
echo "1. Train ES OOF ranker system"
echo "=============================="

if [ "$TRAIN_BEFORE_SUBMISSION" = "1" ]; then
  python -u "$TRAIN_SCRIPT" \
    --train_file "$TRAIN_FILE" \
    --output_resources "$ES_RESOURCES" \
    --output_detector "$ES_DETECTOR" \
    --output_ranker "$ES_RANKER" \
    --detector_threshold "$DETECTOR_THRESHOLD" \
    --detector_n_estimators "$DETECTOR_N_ESTIMATORS" \
    --detector_max_depth "$DETECTOR_MAX_DEPTH" \
    --detector_min_samples_leaf "$DETECTOR_MIN_SAMPLES_LEAF" \
    --detector_min_samples_split "$DETECTOR_MIN_SAMPLES_SPLIT" \
    --detector_max_features "$DETECTOR_MAX_FEATURES" \
    --detector_class_weight "$DETECTOR_CLASS_WEIGHT" \
    --detector_change_weight "$DETECTOR_CHANGE_WEIGHT" \
    --ranker_oof_folds "$RANKER_OOF_FOLDS" \
    --ranker_unchanged_keep_prob "$RANKER_UNCHANGED_KEEP_PROB" \
    --ranker_n_estimators "$RANKER_N_ESTIMATORS" \
    --ranker_max_depth "$RANKER_MAX_DEPTH" \
    --ranker_min_samples_leaf "$RANKER_MIN_SAMPLES_LEAF" \
    --ranker_min_samples_split "$RANKER_MIN_SAMPLES_SPLIT" \
    --ranker_max_features "$RANKER_MAX_FEATURES" \
    --ranker_class_weight "$RANKER_CLASS_WEIGHT" \
    --ranker_positive_weight "$RANKER_POSITIVE_WEIGHT" \
    --ranker_threshold "$RANKER_THRESHOLD" \
    --candidate_margin "$CANDIDATE_MARGIN" \
    --changed_positive_weight "$CHANGED_POSITIVE_WEIGHT" \
    --changed_copy_wrong_weight "$CHANGED_COPY_WRONG_WEIGHT" \
    --changed_wrong_weight "$CHANGED_WRONG_WEIGHT" \
    --unchanged_copy_positive_weight "$UNCHANGED_COPY_POSITIVE_WEIGHT" \
    --overchange_negative_weight "$OVERCHANGE_NEGATIVE_WEIGHT"
else
  echo "Skip training because TRAIN_BEFORE_SUBMISSION=0"
fi

echo "=============================="
echo "2. Check trained files"
echo "=============================="

if [ ! -f "$ES_DETECTOR" ]; then
  echo "ERROR: missing detector: $ES_DETECTOR"
  exit 1
fi

if [ ! -f "$ES_RANKER" ]; then
  echo "ERROR: missing OOF ranker: $ES_RANKER"
  exit 1
fi

if [ ! -f "$ES_RESOURCES" ]; then
  echo "ERROR: missing resources: $ES_RESOURCES"
  exit 1
fi

echo "=============================="
echo "3. Remove previous submission"
echo "=============================="

rm -f "$OUTPUT_JSON"
rm -f "$OUTPUT_ZIP"

echo "=============================="
echo "4. Make submission.zip"
echo "=============================="

python -u "$SUBMISSION_SCRIPT" \
  --es_model_dir "$ES_MODEL_DIR" \
  --es_detector "$ES_DETECTOR" \
  --es_ranker "$ES_RANKER" \
  --es_resources "$ES_RESOURCES" \
  --output_json "$OUTPUT_JSON" \
  --output_zip "$OUTPUT_ZIP" \
  --detector_threshold "$DETECTOR_THRESHOLD" \
  --candidate_margin "$CANDIDATE_MARGIN" \
  --byt5_copy_detector_threshold "$BYT5_COPY_DETECTOR_THRESHOLD" \
  --byt5_candidate_margin "$BYT5_CANDIDATE_MARGIN" \
  --use_byt5 \
  --num_beams "$NUM_BEAMS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --byt5_generate_batch_size "$BYT5_GENERATE_BATCH_SIZE"

echo "=============================="
echo "5. Check output"
echo "=============================="

if [ ! -f "$OUTPUT_ZIP" ]; then
  echo "ERROR: submission.zip was not created."
  exit 1
fi

if [ ! -f "$OUTPUT_JSON" ]; then
  echo "ERROR: predictions.json was not created."
  exit 1
fi

ls -lh "$OUTPUT_ZIP"
ls -lh "$OUTPUT_JSON"

echo "=============================="
echo "Done."
echo "Submission file:"
echo "  $OUTPUT_ZIP"
echo "=============================="