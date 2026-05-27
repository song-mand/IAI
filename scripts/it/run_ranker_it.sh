#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

mkdir -p scripts/it/ranker_artifacts
mkdir -p scripts/it/ranker_outputs
mkdir -p scripts/it/logs

echo "=================================================="
echo "1. Train IT candidate ranker"
echo "=================================================="

python scripts/it/train_ranker_it.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --lang it \
  --kfold 5 \
  --seed 42 \
  --out-dir scripts/it/ranker_artifacts \
  2>&1 | tee scripts/it/logs/train_ranker_it.log

echo "=================================================="
echo "2. Make IT ranker submission"
echo "=================================================="

python scripts/it/sub_ranker_it.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --model scripts/it/ranker_artifacts/it_ranker.pkl \
  --out-dir scripts/it/ranker_outputs \
  2>&1 | tee scripts/it/logs/sub_ranker_it.log

echo "=================================================="
echo "3. Done"
echo "=================================================="

python - <<'PY'
import zipfile

path = "scripts/it/ranker_outputs/submission.zip"
with zipfile.ZipFile(path, "r") as zf:
    print("zip content:")
    for info in zf.infolist():
        print(f"- {info.filename} ({info.file_size} bytes)")
PY