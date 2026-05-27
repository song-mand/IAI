#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

# Use after final models are trained by train_final_models.py train_final.
# Creates ./submission.zip.

python scripts/sub_hybrid_subfinal.py \
  --final_model_dir ./final_model \
  --detector_dir ./detectors \
  --output_dir ./submission_files \
  --zip_path ./submission.zip \
  --es_threshold 0.48 \
  --it_threshold 0.55 \
  --mfr_min_conf 0.65 
