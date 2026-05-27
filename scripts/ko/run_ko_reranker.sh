#!/usr/bin/env bash
# Korean contextual MFR reranker runner.
# Put this file at: scripts/ko/run_ko_reranker.sh
# Run from project root or anywhere:
#   bash scripts/ko/run_ko_reranker.sh eval
#   bash scripts/ko/run_ko_reranker.sh final
#   bash scripts/ko/run_ko_reranker.sh predict
#
# Modes:
#   eval    : train on train split, evaluate on validation split, save model
#   final   : train on train+validation, then predict Korean test rows
#   predict : use an existing saved model and predict Korean test rows only
#   zip     : same as predict, but also create a zip containing ko_predictions.json
#
# Optional environment variables:
#   PYTHON=python3
#   TRAIN_PATH=data/train-00000-of-00001.parquet
#   VALID_PATH=data/validation-00000-of-00001.parquet
#   TEST_PATH=data/test-00000-of-00001.parquet
#   MODEL_DIR=artifacts/ko_reranker
#   OUTPUT_JSON=submission_files/ko_predictions.json
#   ZIP_PATH=submission_files/ko_predictions.zip

set -euo pipefail

MODE="${1:-final}"
PYTHON_BIN="${PYTHON:-python}"

# Resolve project root from this script path: scripts/ko/run_ko_reranker.sh -> project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

choose_path() {
  # Usage: choose_path ENV_VALUE candidate1 candidate2 ...
  local env_value="$1"
  shift

  if [[ -n "${env_value}" ]]; then
    echo "${env_value}"
    return 0
  fi

  local p
  for p in "$@"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done

  # Return empty string. Python scripts will then use HuggingFace dataset fallback.
  echo ""
}

TRAIN_PATH_RESOLVED="$(choose_path "${TRAIN_PATH:-}" \
  "data/train-00000-of-00001.parquet" \
  "train-00000-of-00001.parquet")"

VALID_PATH_RESOLVED="$(choose_path "${VALID_PATH:-}" \
  "data/validation-00000-of-00001.parquet" \
  "validation-00000-of-00001.parquet")"

TEST_PATH_RESOLVED="$(choose_path "${TEST_PATH:-}" \
  "data/test-00000-of-00001.parquet" \
  "test-00000-of-00001.parquet")"

MODEL_DIR="${MODEL_DIR:-artifacts/ko_reranker}"
OUTPUT_JSON="${OUTPUT_JSON:-submission_files/ko_predictions.json}"
ZIP_PATH="${ZIP_PATH:-submission_files/ko_predictions.zip}"

print_config() {
  echo "============================================================"
  echo "Korean reranker runner"
  echo "============================================================"
  echo "mode       : ${MODE}"
  echo "python     : ${PYTHON_BIN}"
  echo "root       : ${PROJECT_ROOT}"
  echo "train path : ${TRAIN_PATH_RESOLVED:-HF dataset fallback}"
  echo "valid path : ${VALID_PATH_RESOLVED:-HF dataset fallback}"
  echo "test path  : ${TEST_PATH_RESOLVED:-HF dataset fallback}"
  echo "model dir  : ${MODEL_DIR}"
  echo "output json: ${OUTPUT_JSON}"
  echo "zip path   : ${ZIP_PATH}"
  echo "============================================================"
}

check_python_packages() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ["pandas", "pyarrow", "sklearn", "joblib", "tqdm"]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]

if missing:
    print("Missing packages:", ", ".join(missing))
    print("Install with:")
    print("  pip install pandas pyarrow scikit-learn joblib tqdm")
    print("If you use HuggingFace fallback without local parquet files, also install:")
    print("  pip install datasets")
    sys.exit(1)
PY
}

make_train_args() {
  local args=("scripts/ko/train_ko_reranker.py" "--model-dir" "${MODEL_DIR}")

  if [[ -n "${TRAIN_PATH_RESOLVED}" ]]; then
    args+=("--train-path" "${TRAIN_PATH_RESOLVED}")
  fi

  if [[ -n "${VALID_PATH_RESOLVED}" ]]; then
    args+=("--valid-path" "${VALID_PATH_RESOLVED}")
  fi

  printf '%q ' "${args[@]}"
}

make_predict_args() {
  local args=("scripts/ko/sub_ko_reranker.py" "--model-dir" "${MODEL_DIR}" "--output-json" "${OUTPUT_JSON}")

  if [[ -n "${TEST_PATH_RESOLVED}" ]]; then
    args+=("--test-path" "${TEST_PATH_RESOLVED}")
  fi

  printf '%q ' "${args[@]}"
}

run_eval() {
  echo "[1/1] Train on train split and evaluate on validation split"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(make_train_args)
}

run_final() {
  echo "[1/2] Fit final Korean reranker with train + validation"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(make_train_args) --fit-final-with-valid

  echo "[2/2] Predict Korean test rows"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(make_predict_args)
}

run_predict() {
  echo "[1/1] Predict Korean test rows with existing model"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(make_predict_args)
}

run_zip() {
  echo "[1/1] Predict Korean test rows and create zip"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(make_predict_args) --zip-path "${ZIP_PATH}"
}

print_config
check_python_packages

case "${MODE}" in
  eval)
    run_eval
    ;;
  final)
    run_final
    ;;
  predict)
    run_predict
    ;;
  zip)
    run_zip
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: bash scripts/ko/run_ko_reranker.sh [eval|final|predict|zip]" >&2
    exit 2
    ;;
esac

echo "Done."
