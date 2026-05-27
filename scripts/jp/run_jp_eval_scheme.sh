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
check_file TRAIN "$TRAIN_PARQUET"
check_file VALID "$VALID_PARQUET"
JP_LANG_CODE=${JP_LANG_CODE:-ja}
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/final_model/jp_scheme_byt5}
ARTIFACT_PATH=${ARTIFACT_PATH:-$REPO_ROOT/final_model/jp_scheme_artifacts/jp_scheme_artifacts.json}
JP_BATCH_SIZE=${JP_BATCH_SIZE:-32}
NO_BYT5=${NO_BYT5:-0}

EXTRA_ARGS=()
if [[ "$NO_BYT5" == "1" ]]; then
  EXTRA_ARGS+=(--no_byt5)
fi
if [[ "${REBUILD_ARTIFACTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--rebuild_artifacts)
fi

python "$SCRIPT_DIR/eval_jp_scheme.py" \
  --train_parquet "$TRAIN_PARQUET" \
  --validation_parquet "$VALID_PARQUET" \
  --model_dir "$MODEL_DIR" \
  --artifact_path "$ARTIFACT_PATH" \
  --batch_size "$JP_BATCH_SIZE" \
  --lang_code "$JP_LANG_CODE" \
  "${EXTRA_ARGS[@]}"
