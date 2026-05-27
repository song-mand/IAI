#!/usr/bin/env bash
# Run from project root:
#   bash scripts/ko/run_same_format_submission.sh final
#   bash scripts/ko/run_same_format_submission.sh predict
#
# This runner creates submission.zip using the SAME output order/format as old sub.py.

set -euo pipefail

MODE="${1:-final}"
PYTHON_BIN="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_PATH="${TRAIN_PATH:-data/train-00000-of-00001.parquet}"
VALID_PATH="${VALID_PATH:-data/validation-00000-of-00001.parquet}"
KO_MODEL_DIR="${KO_MODEL_DIR:-artifacts/ko_reranker}"
SEQ2SEQ_MODEL_ROOT="${SEQ2SEQ_MODEL_ROOT:-final_model}"
OUTPUT_JSON="${OUTPUT_JSON:-submission_files/predictions.json}"
ZIP_PATH="${ZIP_PATH:-submission.zip}"
HF_DATASET="${HF_DATASET:-weerayut/multilexnorm2026-dev-pub}"

use_path_arg() {
  local flag="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    printf '%q %q ' "${flag}" "${path}"
  fi
}

check_packages() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
required = ["datasets", "torch", "transformers", "tqdm", "sklearn", "joblib"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    print("Install roughly with:")
    print("  pip install datasets torch transformers tqdm scikit-learn joblib pandas pyarrow")
    sys.exit(1)
PY
}

train_final() {
  local args=("scripts/ko/train_ko_reranker.py" "--model-dir" "${KO_MODEL_DIR}" "--fit-final-with-valid")
  if [[ -f "${TRAIN_PATH}" ]]; then
    args+=("--train-path" "${TRAIN_PATH}")
  fi
  if [[ -f "${VALID_PATH}" ]]; then
    args+=("--valid-path" "${VALID_PATH}")
  fi
  "${PYTHON_BIN}" "${args[@]}"
}

predict_same_format() {
  "${PYTHON_BIN}" scripts/ko/sub_same_format_with_ko_reranker.py \
    --hf-dataset "${HF_DATASET}" \
    --ko-model-dir "${KO_MODEL_DIR}" \
    --seq2seq-model-root "${SEQ2SEQ_MODEL_ROOT}" \
    --output-json "${OUTPUT_JSON}" \
    --zip-path "${ZIP_PATH}"
}

check_packages

echo "============================================================"
echo "Same-format submission runner"
echo "============================================================"
echo "mode        : ${MODE}"
echo "root        : ${PROJECT_ROOT}"
echo "ko model dir: ${KO_MODEL_DIR}"
echo "output zip  : ${ZIP_PATH}"
echo "============================================================"

case "${MODE}" in
  final)
    echo "[1/2] Train KO reranker with train + validation"
    train_final
    echo "[2/2] Create submission.zip in old sub.py format"
    predict_same_format
    ;;
  predict)
    echo "[1/1] Create submission.zip in old sub.py format with existing KO reranker"
    predict_same_format
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: bash scripts/ko/run_same_format_submission.sh [final|predict]" >&2
    exit 2
    ;;
esac

# Quick structural check: predictions.json exists inside zip.
"${PYTHON_BIN}" - <<PY
import json, zipfile
zip_path = ${ZIP_PATH@Q}
with zipfile.ZipFile(zip_path) as zf:
    assert "predictions.json" in zf.namelist(), zf.namelist()
    data = json.loads(zf.read("predictions.json").decode("utf-8"))
print(f"OK: {zip_path} contains predictions.json with {len(data)} rows")
print("First row keys:", sorted(data[0].keys()) if data else [])
PY
