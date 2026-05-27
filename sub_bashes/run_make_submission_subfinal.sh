#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

# ============================================================
# run_make_submission.sh
#
# Purpose:
#   1. Train final/subfinal models on the FULL training data.
#   2. Create submission.zip using sub_hybrid_final.py.
#
# Important:
#   This script does NOT split the training data.
#   It uses the full ./data/train-00000-of-00001.parquet for training.
# ============================================================

# ------------------------------
# Path settings
# ------------------------------
TRAIN_FILE="./data/train-00000-of-00001.parquet"

TRAIN_SCRIPT="scripts/train_all_esit_models.py"
SUBMISSION_SCRIPT="scripts/sub_hybrid_subfinal.py"

FINAL_MODEL_DIR="./final_model"
DETECTOR_DIR="./detectors"

SUBMISSION_DIR="./submission_files"
ZIP_PATH="./submission.zip"

# ------------------------------
# Train settings
# ------------------------------
TRAIN_SEED=8
BATCH_SIZE=16

# Set this to 0 if models are already trained and you only want to make submission.zip.
TRAIN_BEFORE_SUBMISSION=1

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
  echo "Expected location:"
  echo "  ~/iai_code/scripts/train_all_esit_models.py"
  exit 1
fi

if [ ! -f "$SUBMISSION_SCRIPT" ]; then
  echo "ERROR: SUBMISSION_SCRIPT not found: $SUBMISSION_SCRIPT"
  echo "Expected location:"
  echo "  ~/iai_code/scripts/sub_hybrid_subfinal.py"
  exit 1
fi

mkdir -p "$FINAL_MODEL_DIR"
mkdir -p "$DETECTOR_DIR"
mkdir -p "$SUBMISSION_DIR"

echo "=============================="
echo "1. Train final/subfinal models on FULL train data"
echo "=============================="

if [ "$TRAIN_BEFORE_SUBMISSION" = "1" ]; then
  python "$TRAIN_SCRIPT" \
    --train_file "$TRAIN_FILE" \
    --final_model_dir "$FINAL_MODEL_DIR" \
    --detector_dir "$DETECTOR_DIR" \
    --seed "$TRAIN_SEED" \
    --batch_size "$BATCH_SIZE"
else
  echo "Skip training because TRAIN_BEFORE_SUBMISSION=0"
fi

echo "=============================="
echo "2. Check trained model files"
echo "=============================="

missing=0

for lang in es it; do
  if [ ! -d "${FINAL_MODEL_DIR}/${lang}_model" ]; then
    echo "MISSING: ${FINAL_MODEL_DIR}/${lang}_model"
    missing=1
  fi

  if [ ! -f "${DETECTOR_DIR}/${lang}_change_detector_rf.joblib" ]; then
    echo "MISSING: ${DETECTOR_DIR}/${lang}_change_detector_rf.joblib"
    missing=1
  fi
done

if [ "$missing" = "1" ]; then
  echo "ERROR: Required ES/IT model or detector files are missing."
  exit 1
fi

echo "=============================="
echo "3. Remove previous submission files"
echo "=============================="

rm -rf "$SUBMISSION_DIR"
rm -f "$ZIP_PATH"
mkdir -p "$SUBMISSION_DIR"

echo "=============================="
echo "4. Make submission.zip"
echo "=============================="

# Current sub_hybrid_final.py uses hardcoded paths:
#   ./final_model/{lang}_model
#   ./detectors/{lang}_change_detector_rf.joblib
#   ./submission_files/predictions.json
#   ./submission.zip
#
# So do not pass command-line arguments unless your local sub_hybrid_final.py supports them.
python "$SUBMISSION_SCRIPT"

echo "=============================="
echo "5. Check output"
echo "=============================="

if [ ! -f "$ZIP_PATH" ]; then
  echo "ERROR: submission.zip was not created."
  exit 1
fi

if [ ! -f "${SUBMISSION_DIR}/predictions.json" ]; then
  echo "ERROR: predictions.json was not created."
  exit 1
fi

ls -lh "$ZIP_PATH"
ls -lh "${SUBMISSION_DIR}/predictions.json"

echo "=============================="
echo "Done."
echo "Submission file:"
echo "  $ZIP_PATH"
echo "=============================="
