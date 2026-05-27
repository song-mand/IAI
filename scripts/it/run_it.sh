#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p scripts/it/artifacts
mkdir -p scripts/it/logs
mkdir -p submission_files

echo "========================================"
echo "1. Train / build IT normalization artifact"
echo "========================================"

python3 scripts/it/train_it.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --train-split train \
  --lang it \
  --out-dir scripts/it/artifacts \
  --dev-ratio 0.15 \
  --seed 42 \
  2>&1 | tee scripts/it/logs/train_it.log

echo "========================================"
echo "2. Make submission.zip with IT normalizer"
echo "========================================"

python3 scripts/it/sub_it.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --train-split train \
  --eval-split test \
  --artifact scripts/it/artifacts/it_norm_artifacts.json \
  --out-dir submission_files \
  --zip-name submission.zip \
  2>&1 | tee scripts/it/logs/sub_it.log

echo "========================================"
echo "3. Check zip"
echo "========================================"

python3 - <<'PY'
import zipfile

zip_path = "submission.zip"

with zipfile.ZipFile(zip_path, "r") as zf:
    print("zip content:")
    for name in zf.namelist():
        info = zf.getinfo(name)
        print(f"- {name} ({info.file_size} bytes)")

    assert "predictions.json" in zf.namelist(), "predictions.json not found"

print("zip check OK")
PY

echo "Done."