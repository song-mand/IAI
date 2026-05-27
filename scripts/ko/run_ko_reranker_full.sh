#!/usr/bin/env bash
# Full all-language runner with Korean contextual reranker.
# Put this file at: scripts/ko/run_ko_reranker_full.sh
# Run from project root:
#   bash scripts/ko/run_ko_reranker_full.sh eval
#   bash scripts/ko/run_ko_reranker_full.sh valid-pred
#   bash scripts/ko/run_ko_reranker_full.sh final
#   bash scripts/ko/run_ko_reranker_full.sh predict-all
#   bash scripts/ko/run_ko_reranker_full.sh check-final
#
# Modes:
#   eval        : train KO reranker on train, evaluate on validation
#   valid-pred  : train KO reranker on train only, predict VALIDATION rows, zip valid_predictions.zip
#                 Use this when running local scoring.py with validation labels.
#   final       : train KO reranker on train+validation, predict TEST rows, zip submission.zip
#                 Use this for final Codabench submission.
#   predict-all : use existing KO model, predict TEST rows, zip submission.zip
#   check-final : check whether submission.zip raw rows match TEST_PATH
#
# Optional environment variables:
#   PYTHON=python3
#   TRAIN_PATH=data/train-00000-of-00001.parquet
#   VALID_PATH=data/validation-00000-of-00001.parquet
#   TEST_PATH=data/test-00000-of-00001.parquet
#   KO_MODEL_DIR=artifacts/ko_reranker
#   SEQ2SEQ_MODEL_ROOT=final_model
#   OUTPUT_JSON=submission_files/predictions.json
#   ZIP_PATH=submission.zip
#   VALID_OUTPUT_JSON=submission_files/valid_predictions.json
#   VALID_ZIP_PATH=valid_predictions.zip

set -euo pipefail

MODE="${1:-final}"
PYTHON_BIN="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

choose_path() {
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

KO_MODEL_DIR="${KO_MODEL_DIR:-artifacts/ko_reranker}"
SEQ2SEQ_MODEL_ROOT="${SEQ2SEQ_MODEL_ROOT:-final_model}"
OUTPUT_JSON="${OUTPUT_JSON:-submission_files/predictions.json}"
ZIP_PATH="${ZIP_PATH:-submission.zip}"
VALID_OUTPUT_JSON="${VALID_OUTPUT_JSON:-submission_files/valid_predictions.json}"
VALID_ZIP_PATH="${VALID_ZIP_PATH:-valid_predictions.zip}"

print_config() {
  echo "============================================================"
  echo "Full runner with Korean reranker"
  echo "============================================================"
  echo "mode              : ${MODE}"
  echo "python            : ${PYTHON_BIN}"
  echo "root              : ${PROJECT_ROOT}"
  echo "train path        : ${TRAIN_PATH_RESOLVED:-HF dataset fallback}"
  echo "valid path        : ${VALID_PATH_RESOLVED:-none/HF test fallback where applicable}"
  echo "test path         : ${TEST_PATH_RESOLVED:-HF dataset fallback}"
  echo "ko model dir      : ${KO_MODEL_DIR}"
  echo "seq2seq model root: ${SEQ2SEQ_MODEL_ROOT}"
  echo "output json       : ${OUTPUT_JSON}"
  echo "zip path          : ${ZIP_PATH}"
  echo "valid output json : ${VALID_OUTPUT_JSON}"
  echo "valid zip path    : ${VALID_ZIP_PATH}"
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

train_args() {
  local args=("scripts/ko/train_ko_reranker.py" "--model-dir" "${KO_MODEL_DIR}")
  if [[ -n "${TRAIN_PATH_RESOLVED}" ]]; then
    args+=("--train-path" "${TRAIN_PATH_RESOLVED}")
  fi
  if [[ -n "${VALID_PATH_RESOLVED}" ]]; then
    args+=("--valid-path" "${VALID_PATH_RESOLVED}")
  fi
  printf '%q ' "${args[@]}"
}

predict_target_args() {
  local target_path="$1"
  local output_json="$2"
  local zip_path="$3"
  local use_valid_flag="$4"  # yes/no

  local args=("scripts/ko/sub_all_with_ko_reranker.py" \
    "--ko-model-dir" "${KO_MODEL_DIR}" \
    "--seq2seq-model-root" "${SEQ2SEQ_MODEL_ROOT}" \
    "--output-json" "${output_json}" \
    "--zip-path" "${zip_path}")

  if [[ -n "${TRAIN_PATH_RESOLVED}" ]]; then
    args+=("--train-path" "${TRAIN_PATH_RESOLVED}")
  fi
  if [[ -n "${VALID_PATH_RESOLVED}" ]]; then
    args+=("--valid-path" "${VALID_PATH_RESOLVED}")
  fi
  if [[ -n "${target_path}" ]]; then
    args+=("--test-path" "${target_path}" "--reference-path" "${target_path}")
  fi
  if [[ "${use_valid_flag}" == "no" ]]; then
    args+=("--no-use-valid-for-mfr")
  else
    args+=("--use-valid-for-mfr")
  fi

  printf '%q ' "${args[@]}"
}

run_eval() {
  echo "[1/1] Train KO reranker on train and evaluate on validation"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(train_args)
}

run_valid_pred() {
  if [[ -z "${VALID_PATH_RESOLVED}" ]]; then
    echo "valid-pred requires VALID_PATH or data/validation-00000-of-00001.parquet" >&2
    exit 1
  fi
  echo "[1/2] Train KO reranker on train only"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(train_args)

  echo "[2/2] Predict VALIDATION rows. Use ${VALID_ZIP_PATH} for local validation scoring."
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(predict_target_args "${VALID_PATH_RESOLVED}" "${VALID_OUTPUT_JSON}" "${VALID_ZIP_PATH}" "no")
}

run_final() {
  if [[ -z "${TEST_PATH_RESOLVED}" ]]; then
    echo "final requires TEST_PATH or data/test-00000-of-00001.parquet" >&2
    exit 1
  fi
  echo "[1/2] Fit final KO reranker with train + validation"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(train_args) --fit-final-with-valid

  echo "[2/2] Predict TEST rows and create final submission.zip"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(predict_target_args "${TEST_PATH_RESOLVED}" "${OUTPUT_JSON}" "${ZIP_PATH}" "yes")
}

run_predict_all() {
  if [[ -z "${TEST_PATH_RESOLVED}" ]]; then
    echo "predict-all requires TEST_PATH or data/test-00000-of-00001.parquet" >&2
    exit 1
  fi
  echo "[1/1] Predict TEST rows with existing KO model"
  # shellcheck disable=SC2046
  "${PYTHON_BIN}" $(predict_target_args "${TEST_PATH_RESOLVED}" "${OUTPUT_JSON}" "${ZIP_PATH}" "yes")
}

run_check_final() {
  if [[ -z "${TEST_PATH_RESOLVED}" ]]; then
    echo "check-final requires TEST_PATH or data/test-00000-of-00001.parquet" >&2
    exit 1
  fi
  "${PYTHON_BIN}" scripts/ko/check_submission_raw_order.py \
    --label-path "${TEST_PATH_RESOLVED}" \
    --pred-zip "${ZIP_PATH}"
}

print_config
check_python_packages

case "${MODE}" in
  eval)
    run_eval
    ;;
  valid-pred)
    run_valid_pred
    ;;
  final)
    run_final
    ;;
  predict-all)
    run_predict_all
    ;;
  check-final)
    run_check_final
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: bash scripts/ko/run_ko_reranker_full.sh [eval|valid-pred|final|predict-all|check-final]" >&2
    exit 2
    ;;
esac

echo "Done."
