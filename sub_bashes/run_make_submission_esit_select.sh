#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

# ============================================================
# run_make_submission_selected.sh
#
# Purpose:
#   1. Train only selected ES/IT language(s).
#   2. Use existing models for all other languages.
#   3. Create submission.zip using sub_hybrid_subfinal.py.
#
# This script does NOT split the training data.
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

# Set this to 1 if you want to train selected ES/IT language(s).
# Set this to 0 if all models are already trained and you only want submission.zip.
TRAIN_BEFORE_SUBMISSION=1

# Train only selected ES/IT language(s).
#
# Options:
#   TRAIN_LANGS="es"
#   TRAIN_LANGS="it"
#   TRAIN_LANGS="es,it"
#
# Example:
#   If TRAIN_LANGS="es", only es_model and es_change_detector_rf.joblib are updated.
#   Existing it/western models are kept.
TRAIN_LANGS="es"

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
echo "1. Train selected ES/IT model(s)"
echo "=============================="

if [ "$TRAIN_BEFORE_SUBMISSION" = "1" ]; then
  if [ -z "$TRAIN_LANGS" ]; then
    echo "ERROR: TRAIN_LANGS is empty."
    echo "Use TRAIN_LANGS=\"es\", \"it\", or \"es,it\"."
    exit 1
  fi

  echo "Training only: $TRAIN_LANGS"
  echo "Western languages will NOT be retrained."

  python "$TRAIN_SCRIPT" \
    --train_file "$TRAIN_FILE" \
    --final_model_dir "$FINAL_MODEL_DIR" \
    --detector_dir "$DETECTOR_DIR" \
    --seed "$TRAIN_SEED" \
    --batch_size "$BATCH_SIZE" \
    --no_train_western \
    --es_it_langs "$TRAIN_LANGS"
else
  echo "Skip training because TRAIN_BEFORE_SUBMISSION=0"
fi

echo "=============================="
echo "2. Check required ES/IT model files"
echo "=============================="

missing=0

# For submission, both ES and IT hybrid resources should exist.
# If you trained only one of them, the other one must already exist.
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
  echo "If you trained only ES, make sure existing IT files are already present."
  echo "If you trained only IT, make sure existing ES files are already present."
  exit 1
fi

echo "=============================="
echo "3. Optional check for existing western models"
echo "=============================="

for lang in en da de hr nl sl sr tr iden trde; do
  if [ -d "${FINAL_MODEL_DIR}/${lang}_model" ]; then
    echo "${lang}: existing ByT5 model found"
  else
    echo "${lang}: no ByT5 model found; submission script may use MFR fallback if supported"
  fi
done

echo "=============================="
echo "4. Remove previous submission files"
echo "=============================="

rm -rf "$SUBMISSION_DIR"
rm -f "$ZIP_PATH"
mkdir -p "$SUBMISSION_DIR"

echo "=============================="
echo "5. Make submission.zip"
echo "=============================="

python "$SUBMISSION_SCRIPT"

echo "=============================="
echo "6. Check output"
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