#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

TRAIN_FILE="./data/train-00000-of-00001.parquet"
ES_MODEL_DIR="./final_model/es_model"
ES_DETECTOR="./detectors/es_change_detector_rf.joblib"
ES_RANKER="./detectors/es_candidate_ranker_rf.joblib"
ES_RESOURCES="./detectors/es_resources.joblib"

SEED=5
BATCH_SIZE=16

# Safe setting
ES_EPOCHS=5
ES_UNCHANGED_KEEP_PROB=0.8
ES_CHANGED_REPEAT=3

# Aggressive candidates:
# ES_EPOCHS=3
# ES_UNCHANGED_KEEP_PROB=0.2
# ES_CHANGED_REPEAT=4

DETECTOR_THRESHOLD=0.40
RANKER_THRESHOLD=0.25
MFR_MIN_CONF=0.65

python - <<'PY'
import torch, sklearn, joblib, transformers, datasets, peft
print("packages OK")
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python -u scripts/es/train_es_hybrid_strong.py \
  --train_file "$TRAIN_FILE" \
  --output_model_dir "$ES_MODEL_DIR" \
  --output_detector "$ES_DETECTOR" \
  --output_ranker "$ES_RANKER" \
  --output_resources "$ES_RESOURCES" \
  --seed "$SEED" \
  --epochs "$ES_EPOCHS" \
  --unchanged_keep_prob "$ES_UNCHANGED_KEEP_PROB" \
  --changed_repeat "$ES_CHANGED_REPEAT" \
  --batch_size "$BATCH_SIZE" \
  --detector_threshold "$DETECTOR_THRESHOLD" \
  --ranker_threshold "$RANKER_THRESHOLD" \
  --valid_ratio 0.0

python -u scripts/es/sub_es_hybrid_all_mfr.py \
  --es_model_dir "$ES_MODEL_DIR" \
  --es_detector "$ES_DETECTOR" \
  --es_ranker "$ES_RANKER" \
  --es_resources "$ES_RESOURCES" \
  --detector_threshold "$DETECTOR_THRESHOLD" \
  --ranker_threshold "$RANKER_THRESHOLD" \
  --mfr_min_conf "$MFR_MIN_CONF" \
  --use_byt5 \
  --low_ranker_byt5_threshold 0.90\
  --num_beams 1 \
  --max_new_tokens 12 \
  --byt5_generate_batch_size 64

ls -lh submission.zip

:<<'END'
cd ~/iai_code
source .venv/bin/activate

python -u scripts/es/sub_es_hybrid_all_mfr.py \
  --es_model_dir ./final_model/es_model \
  --es_detector ./detectors/es_change_detector_rf.joblib \
  --es_ranker ./detectors/es_candidate_ranker_rf.joblib \
  --es_resources ./detectors/es_resources.joblib \
  --detector_threshold 0.40 \
  --ranker_threshold 0.20 \ 
  --mfr_min_conf 1.01 \
  --use_byt5 \
  --low_ranker_byt5_threshold 0.90 \
  --num_beams 1 \
  --max_new_tokens 12 \
  --byt5_generate_batch_size 64
END
cd ~/iai_code
source .venv/bin/activate
:<<'END'
python -u scripts/es/sub_es_hybrid_all_mfr.py \
  --es_model_dir ./final_model/es_model \
  --es_detector ./detectors/es_change_detector_rf.joblib \
  --es_ranker ./detectors/es_candidate_ranker_rf.joblib \
  --es_resources ./detectors/es_resources.joblib \
  --detector_threshold 0.43 \
  --ranker_threshold 0.25 \
  --mfr_min_conf 0.68 \
  --use_byt5 \
  --low_ranker_byt5_threshold 0.90 \
  --num_beams 1 \
  --max_new_tokens 12 \
  --byt5_generate_batch_size 64
END