#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

mkdir -p scripts/it/monoise_outputs
mkdir -p scripts/it/logs

MODE="${1:-m2_mfr_preserve_unseen_safe}"
RUN_MONOISE="${2:-1}"

echo "=================================================="
echo "Run MoNoise-gated IT submission"
echo "MODE=$MODE"
echo "RUN_MONOISE=$RUN_MONOISE"
echo "=================================================="

RUN_FLAG=""
if [ "$RUN_MONOISE" = "1" ]; then
  RUN_FLAG="--run-monoise"
fi

python scripts/it/sub_monoise_it.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --mode "$MODE" \
  --out-dir scripts/it/monoise_outputs \
  --zip-name "submission_${MODE}.zip" \
  --debug-it \
  $RUN_FLAG \
  2>&1 | tee "scripts/it/logs/sub_monoise_${MODE}.log"

echo "=================================================="
echo "Done"
echo "=================================================="

python - <<PY
import zipfile
path = "scripts/it/monoise_outputs/submission_${MODE}.zip"
with zipfile.ZipFile(path, "r") as zf:
    print("zip content:")
    for info in zf.infolist():
        print(f"- {info.filename} ({info.file_size} bytes)")
PY