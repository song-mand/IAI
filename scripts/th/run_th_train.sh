#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Thai lexical normalization training pipeline
# - Resource build: MFR, error_dict, correct_dict
# - Detector: RandomForest should-change classifier
# - Candidate generator: ByT5 with Thai synthetic-noise pretrain + real finetune
# ============================================================

# --------------------------
# Common
# --------------------------
export DATASET=${DATASET:-"weerayut/multilexnorm2026-dev-pub"}
export TRAIN_SPLIT=${TRAIN_SPLIT:-"train"}
export TARGET_LANG=${TARGET_LANG:-"th"}
export SEED=${SEED:-42}

# Output paths
export TH_RESOURCE_PATH=${TH_RESOURCE_PATH:-"models/th/th_resources.pkl"}
export TH_RESOURCE_SUMMARY=${TH_RESOURCE_SUMMARY:-"models/th/th_resources_summary.json"}
export TH_DETECTOR_PATH=${TH_DETECTOR_PATH:-"models/th/th_detector.joblib"}
export TH_DETECTOR_META=${TH_DETECTOR_META:-"models/th/th_detector_meta.json"}
export TH_BYT5_OUTPUT=${TH_BYT5_OUTPUT:-"final_model/th_byt5_candidate"}
export TH_BYT5_WORK=${TH_BYT5_WORK:-"models/th/byt5_work"}

# What to run
export BUILD_RESOURCES=${BUILD_RESOURCES:-1}
export TRAIN_DETECTOR=${TRAIN_DETECTOR:-1}
export TRAIN_BYT5=${TRAIN_BYT5:-1}

# --------------------------
# Resource hyperparameters
# --------------------------
export MIN_COUNT=${MIN_COUNT:-1}
export TOP_N_PRINT=${TOP_N_PRINT:-30}

# --------------------------
# Detector hyperparameters
# --------------------------
export VALID_RATIO=${VALID_RATIO:-0.15}
export NEG_RATIO=${NEG_RATIO:-6}
export MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}        # 0 = no cap
export RF_N_ESTIMATORS=${RF_N_ESTIMATORS:-700}
export RF_MAX_DEPTH=${RF_MAX_DEPTH:-18}                 # use "None" for unlimited
export RF_MIN_SAMPLES_LEAF=${RF_MIN_SAMPLES_LEAF:-2}
export RF_MAX_FEATURES=${RF_MAX_FEATURES:-sqrt}
export RF_CLASS_WEIGHT=${RF_CLASS_WEIGHT:-balanced_subsample}
export THRESHOLDS=${THRESHOLDS:-"0.30,0.40,0.50,0.60,0.70,0.80,0.90"}
export MIN_MFR_CONF=${MIN_MFR_CONF:-0.80}
export MIN_MFR_COUNT=${MIN_MFR_COUNT:-2}
export TRAIN_FINAL_ON_ALL=${TRAIN_FINAL_ON_ALL:-1}

# --------------------------
# ByT5 data hyperparameters
# --------------------------
export BASE_MODEL=${BASE_MODEL:-"google/byt5-small"}
export INPUT_FORMAT=${INPUT_FORMAT:-natural}            # natural or sentinel
export KEEP_UNCHANGED_PROB=${KEEP_UNCHANGED_PROB:-0.08}
export AUG_PER_TOKEN=${AUG_PER_TOKEN:-2}
export MAX_REAL_SAMPLES=${MAX_REAL_SAMPLES:-0}          # 0 = no cap
export MAX_AUG_SAMPLES=${MAX_AUG_SAMPLES:-0}            # 0 = no cap

# Thai synthetic noise probabilities
export P_DELETE=${P_DELETE:-0.03}
export P_INSERT=${P_INSERT:-0.03}
export P_SUBSTITUTE=${P_SUBSTITUTE:-0.03}
export P_REPEAT_MARK=${P_REPEAT_MARK:-0.06}
export P_REPEAT_CHAR=${P_REPEAT_CHAR:-0.02}
export P_SWAP_MARK=${P_SWAP_MARK:-0.04}

# ByT5 training hyperparameters
export PRETRAIN_AUG=${PRETRAIN_AUG:-1}
export FINETUNE_REAL=${FINETUNE_REAL:-1}
export AUG_EPOCHS=${AUG_EPOCHS:-1}
export REAL_EPOCHS=${REAL_EPOCHS:-2}
export AUG_LR=${AUG_LR:-5e-5}
export REAL_LR=${REAL_LR:-3e-5}
export BATCH_SIZE=${BATCH_SIZE:-16}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-160}
export MAX_TARGET_LENGTH=${MAX_TARGET_LENGTH:-64}
export NUM_WORKERS=${NUM_WORKERS:-2}
export FP16=${FP16:-0}
export BF16=${BF16:-1}

# LoRA hyperparameters
export USE_LORA=${USE_LORA:-1}
export LORA_R=${LORA_R:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}

mkdir -p models/th final_model submission_files

echo "============================================================"
echo "0. Check environment"
echo "============================================================"
python - <<'PY'
import sys
print('python:', sys.version)
for pkg in ['datasets', 'sklearn', 'transformers', 'torch']:
    try:
        mod = __import__(pkg)
        print(pkg, 'OK')
    except Exception as e:
        print(pkg, 'MISSING/ERROR:', e)
try:
    import torch
    print('cuda:', torch.cuda.is_available())
    if torch.cuda.is_available(): print('gpu:', torch.cuda.get_device_name(0))
except Exception: pass
PY

if [[ "$BUILD_RESOURCES" == "1" ]]; then
  echo "============================================================"
  echo "1. Build Thai resources"
  echo "============================================================"
  python build_th_resources.py \
    --dataset "$DATASET" \
    --split "$TRAIN_SPLIT" \
    --lang "$TARGET_LANG" \
    --out "$TH_RESOURCE_PATH" \
    --summary-out "$TH_RESOURCE_SUMMARY" \
    --min-count "$MIN_COUNT" \
    --top-n "$TOP_N_PRINT"
fi

if [[ "$TRAIN_DETECTOR" == "1" ]]; then
  echo "============================================================"
  echo "2. Train Thai detector"
  echo "============================================================"
  python train_th_detector.py \
    --dataset "$DATASET" \
    --split "$TRAIN_SPLIT" \
    --lang "$TARGET_LANG" \
    --out "$TH_DETECTOR_PATH" \
    --resource-out "$TH_RESOURCE_PATH" \
    --meta-out "$TH_DETECTOR_META" \
    --seed "$SEED" \
    --valid-ratio "$VALID_RATIO" \
    --neg-ratio "$NEG_RATIO" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --n-estimators "$RF_N_ESTIMATORS" \
    --max-depth "$RF_MAX_DEPTH" \
    --min-samples-leaf "$RF_MIN_SAMPLES_LEAF" \
    --max-features "$RF_MAX_FEATURES" \
    --class-weight "$RF_CLASS_WEIGHT" \
    --thresholds "$THRESHOLDS" \
    --min-mfr-conf "$MIN_MFR_CONF" \
    --min-mfr-count "$MIN_MFR_COUNT" \
    --train-final-on-all "$TRAIN_FINAL_ON_ALL"
fi

if [[ "$TRAIN_BYT5" == "1" ]]; then
  echo "============================================================"
  echo "3. Train Thai ByT5 candidate generator"
  echo "============================================================"
  python train_th_byt5_aug.py \
    --dataset "$DATASET" \
    --split "$TRAIN_SPLIT" \
    --lang "$TARGET_LANG" \
    --base-model "$BASE_MODEL" \
    --output-dir "$TH_BYT5_OUTPUT" \
    --work-dir "$TH_BYT5_WORK" \
    --seed "$SEED" \
    --keep-unchanged-prob "$KEEP_UNCHANGED_PROB" \
    --aug-per-token "$AUG_PER_TOKEN" \
    --max-real-samples "$MAX_REAL_SAMPLES" \
    --max-aug-samples "$MAX_AUG_SAMPLES" \
    --input-format "$INPUT_FORMAT" \
    --p-delete "$P_DELETE" \
    --p-insert "$P_INSERT" \
    --p-substitute "$P_SUBSTITUTE" \
    --p-repeat-mark "$P_REPEAT_MARK" \
    --p-repeat-char "$P_REPEAT_CHAR" \
    --p-swap-mark "$P_SWAP_MARK" \
    --pretrain-aug "$PRETRAIN_AUG" \
    --finetune-real "$FINETUNE_REAL" \
    --aug-epochs "$AUG_EPOCHS" \
    --real-epochs "$REAL_EPOCHS" \
    --aug-lr "$AUG_LR" \
    --real-lr "$REAL_LR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-input-length "$MAX_INPUT_LENGTH" \
    --max-target-length "$MAX_TARGET_LENGTH" \
    --num-workers "$NUM_WORKERS" \
    --fp16 "$FP16" \
    --bf16 "$BF16" \
    --use-lora "$USE_LORA" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT"
fi

echo "============================================================"
echo "DONE"
echo "============================================================"
echo "resources: $TH_RESOURCE_PATH"
echo "detector:  $TH_DETECTOR_PATH"
echo "byt5:      $TH_BYT5_OUTPUT"
