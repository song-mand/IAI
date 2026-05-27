#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

mkdir -p scripts/it/loo_outputs
mkdir -p scripts/it/logs

MODE="${1:-l3_cap_strict}"
#l0_mfr  l4_accent_only  l2_loo_category  l3_cap_strict  l5_high_precision
echo "=================================================="
echo "Run LOO-Reliability IT submission"
echo "MODE=$MODE"
echo "=================================================="

python scripts/it/sub_loo_it.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --mode "$MODE" \
  --out-dir scripts/it/loo_outputs \
  --zip-name "submission_${MODE}.zip" \
  --debug-it \
  2>&1 | tee "scripts/it/logs/sub_loo_${MODE}.log"

echo "=================================================="
echo "Done"
echo "=================================================="

python - <<PY
import zipfile
path = "scripts/it/loo_outputs/submission_${MODE}.zip"
with zipfile.ZipFile(path, "r") as zf:
    print("zip content:")
    for info in zf.infolist():
        print(f"- {info.filename} ({info.file_size} bytes)")
PY