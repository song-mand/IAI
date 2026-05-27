#!/usr/bin/env bash
set -euo pipefail

# This file is intended to live in scripts/jp/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

find_parquet() {
  local filename="$1"
  local provided="${2:-}"
  if [[ -n "$provided" ]]; then
    echo "$provided"
    return 0
  fi
  local candidates=(
    "$REPO_ROOT/$filename"
    "$REPO_ROOT/data/$filename"
    "$REPO_ROOT/datasets/$filename"
    "$REPO_ROOT/input/$filename"
    "$SCRIPT_DIR/$filename"
    "$PWD/$filename"
  )
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  echo "$REPO_ROOT/$filename"
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] $label not found: $path" >&2
    echo "        Put parquet files in repo root, data/, datasets/, input/, scripts/jp/, or set ${label}_PARQUET explicitly." >&2
    exit 1
  fi
}

TRAIN_PARQUET=$(find_parquet "train-00000-of-00001.parquet" "${TRAIN_PARQUET:-}")
VALID_PARQUET=$(find_parquet "validation-00000-of-00001.parquet" "${VALID_PARQUET:-}")
TEST_PARQUET=$(find_parquet "test-00000-of-00001.parquet" "${TEST_PARQUET:-}")
check_file TRAIN "$TRAIN_PARQUET"
check_file VALID "$VALID_PARQUET"
# By default, official submission target is loaded from Hugging Face dataset test split.
# TEST_PARQUET is checked only when USE_LOCAL_TEST=1.
if [[ "${USE_LOCAL_TEST:-0}" == "1" ]]; then
  check_file TEST "$TEST_PARQUET"
fi
JP_LANG_CODE=${JP_LANG_CODE:-ja}
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/final_model/jp_scheme_byt5}
ARTIFACT_PATH=${ARTIFACT_PATH:-$REPO_ROOT/final_model/jp_scheme_artifacts/jp_scheme_artifacts.json}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/submission_files}
OUTPUT_ZIP=${OUTPUT_ZIP:-$REPO_ROOT/submission.zip}
JP_BATCH_SIZE=${JP_BATCH_SIZE:-32}

# For final submission, using public validation labels for MFR/artifacts can improve dictionary coverage.
INCLUDE_VALIDATION_FOR_MFR=${INCLUDE_VALIDATION_FOR_MFR:-1}
REBUILD_ARTIFACTS=${REBUILD_ARTIFACTS:-0}
NO_BYT5=${NO_BYT5:-0}
USE_LOCAL_TEST=${USE_LOCAL_TEST:-0}
HF_DATASET_NAME=${HF_DATASET_NAME:-weerayut/multilexnorm2026-dev-pub}

EXTRA_ARGS=()
if [[ "$INCLUDE_VALIDATION_FOR_MFR" == "1" ]]; then
  EXTRA_ARGS+=(--include_validation_for_mfr)
fi
if [[ "$REBUILD_ARTIFACTS" == "1" ]]; then
  EXTRA_ARGS+=(--rebuild_artifacts)
fi
if [[ "$NO_BYT5" == "1" ]]; then
  EXTRA_ARGS+=(--no_byt5)
fi
if [[ "$USE_LOCAL_TEST" == "1" ]]; then
  EXTRA_ARGS+=(--use_local_test)
fi

python "$SCRIPT_DIR/sub_jp_scheme.py" \
  --train_parquet "$TRAIN_PARQUET" \
  --validation_parquet "$VALID_PARQUET" \
  --test_parquet "$TEST_PARQUET" \
  --dataset_name "$HF_DATASET_NAME" \
  --model_dir "$MODEL_DIR" \
  --artifact_path "$ARTIFACT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --output_zip "$OUTPUT_ZIP" \
  --batch_size "$JP_BATCH_SIZE" \
  --lang_code "$JP_LANG_CODE" \
  --rebuild_artifacts \
  --min_mfr_count 2 \
  --min_mfr_best_prob 0.50 \
  --min_mfr_change_rate 0.50 \
  --max_mfr_entropy 0.80 \
  --num_beams 2\
  "${EXTRA_ARGS[@]}"
