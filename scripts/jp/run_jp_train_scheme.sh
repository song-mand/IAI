#!/usr/bin/env bash
set -euo pipefail

# This file is intended to live in scripts/jp/.
# It can be run either as:
#   bash scripts/jp/run_jp_train_scheme.sh
# or from scripts/jp as:
#   bash run_jp_train_scheme.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Automatically find parquet files from common locations.
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
check_file TRAIN "$TRAIN_PARQUET"
check_file VALID "$VALID_PARQUET"

# Uploaded MultiLexNorm parquet uses lang == "ja" for Japanese.
# Set JP_LANG_CODE=jp only if your local data actually uses jp.
JP_LANG_CODE=${JP_LANG_CODE:-ja}

BASE_MODEL=${BASE_MODEL:-google/byt5-small}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/final_model/jp_scheme_byt5}
ARTIFACT_PATH=${ARTIFACT_PATH:-$REPO_ROOT/final_model/jp_scheme_artifacts/jp_scheme_artifacts.json}

JP_EPOCHS=${JP_EPOCHS:-3}
JP_LR=${JP_LR:-3e-5}
JP_BATCH_SIZE=${JP_BATCH_SIZE:-8}
JP_GRAD_ACCUM=${JP_GRAD_ACCUM:-1}
JP_MAX_INPUT_LENGTH=${JP_MAX_INPUT_LENGTH:-192}
JP_MAX_TARGET_LENGTH=${JP_MAX_TARGET_LENGTH:-64}
JP_UNCHANGED_SAMPLE_RATE=${JP_UNCHANGED_SAMPLE_RATE:-0.08}
JP_PUNCT_UNCHANGED_RATE=${JP_PUNCT_UNCHANGED_RATE:-0.50}

# Set USE_LORA=1 if VRAM is tight or you want faster experiments.
USE_LORA=${USE_LORA:-0}
# Set USE_VALIDATION_FOR_TRAINING=1 only for final submission training, not for validation experiments.
USE_VALIDATION_FOR_TRAINING=${USE_VALIDATION_FOR_TRAINING:-0}

EXTRA_ARGS=()
if [[ "$USE_LORA" == "1" ]]; then
  EXTRA_ARGS+=(--use_lora)
fi
if [[ "$USE_VALIDATION_FOR_TRAINING" == "1" ]]; then
  EXTRA_ARGS+=(--use_validation_for_training)
fi
# ByT5/T5 full fine-tuning is often unstable in fp16.
# Default is fp32 for safety. Try JP_FP16=1 only after the fp32 run is stable.
if [[ "${JP_FP16:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--fp16)
fi
if [[ "${JP_BF16:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi

python "$SCRIPT_DIR/train_jp_scheme_byt5.py" \
  --train_parquet "$TRAIN_PARQUET" \
  --validation_parquet "$VALID_PARQUET" \
  --base_model "$BASE_MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --artifact_path "$ARTIFACT_PATH" \
  --epochs "$JP_EPOCHS" \
  --lr "$JP_LR" \
  --batch_size "$JP_BATCH_SIZE" \
  --grad_accum "$JP_GRAD_ACCUM" \
  --max_input_length "$JP_MAX_INPUT_LENGTH" \
  --max_target_length "$JP_MAX_TARGET_LENGTH" \
  --lang_code "$JP_LANG_CODE" \
  --unchanged_sample_rate "$JP_UNCHANGED_SAMPLE_RATE" \
  --punct_unchanged_rate "$JP_PUNCT_UNCHANGED_RATE" \
  "${EXTRA_ARGS[@]}"
