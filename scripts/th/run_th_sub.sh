#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Thai two-stage submission pipeline
# - TH: detector + MFR/error_dict/rules/ByT5 candidate reranker
# - non-TH: MFR baseline
# ============================================================

# --------------------------
# Common / paths
# --------------------------
export DATASET=${DATASET:-"weerayut/multilexnorm2026-dev-pub"}
export TRAIN_SPLIT=${TRAIN_SPLIT:-"train"}
export EVAL_SPLIT=${EVAL_SPLIT:-"test"}
export ALL_LANGS=${ALL_LANGS:-"en,da,de,es,hr,it,nl,sl,sr,tr,iden,trde,id,ja,ko,th,vi"}
export TARGET_LANG=${TARGET_LANG:-"th"}
export SEED=${SEED:-42}

export TH_RESOURCE_PATH=${TH_RESOURCE_PATH:-"models/th/th_resources.pkl"}
export TH_DETECTOR_PATH=${TH_DETECTOR_PATH:-"models/th/th_detector.joblib"}
export TH_BYT5_MODEL=${TH_BYT5_MODEL:-"final_model/th_byt5_candidate"}
export SUBMISSION_DIR=${SUBMISSION_DIR:-"submission_files"}
export ZIP_PATH=${ZIP_PATH:-"submission.zip"}
export DEBUG_PATH=${DEBUG_PATH:-"submission_files/th_debug_samples.json"}

# --------------------------
# Switches
# --------------------------
export USE_DETECTOR=${USE_DETECTOR:-0}
export USE_BYT5=${USE_BYT5:-1}
export USE_RULES=${USE_RULES:-0}
export USE_ERRDICT=${USE_ERRDICT:-0}
export USE_MFR_CANDIDATE=${USE_MFR_CANDIDATE:-0}

# --------------------------
# Reranker / decision hyperparameters
# --------------------------
# DETECTOR_THRESHOLD=auto uses threshold selected during train_th_detector.py
export DETECTOR_THRESHOLD=${DETECTOR_THRESHOLD:-0.00}
export BYT5_THRESHOLD=${BYT5_THRESHOLD:-0.00}
export MIN_MFR_CONF=${MIN_MFR_CONF:-0.80}
export FORCE_MFR_CONF=${FORCE_MFR_CONF:-1.01}
export MIN_MFR_COUNT=${MIN_MFR_COUNT:-2}
export ACCEPT_SCORE_MIN=${ACCEPT_SCORE_MIN:--4.00}
export MAX_EDIT_RATIO=${MAX_EDIT_RATIO:-10.00}
export BYT5_BONUS=${BYT5_BONUS:-5.00}
export REQUIRE_CORRECT_DICT_FOR_BYT5=${REQUIRE_CORRECT_DICT_FOR_BYT5:-0}
export PROTECT_NONTHAI=${PROTECT_NONTHAI:-1}
export MAX_RULE_CANDIDATES=${MAX_RULE_CANDIDATES:-8}
export MAX_ERRDICT_CANDIDATES=${MAX_ERRDICT_CANDIDATES:-5}

# --------------------------
# ByT5 generation hyperparameters
# --------------------------
export INPUT_FORMAT=${INPUT_FORMAT:-natural}
export MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-160}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
export NUM_BEAMS=${NUM_BEAMS:-1}
export NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES:-1}
export INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-64}

# --------------------------
# Debugging / analysis prints
# --------------------------
export PRINT_EXAMPLES=${PRINT_EXAMPLES:-80}
export PRINT_LANG_METRICS=${PRINT_LANG_METRICS:-1}

mkdir -p "$SUBMISSION_DIR"

echo "============================================================"
echo "Make two-stage Thai submission"
echo "============================================================"
python sub_th_two_stage.py \
  --dataset "$DATASET" \
  --train-split "$TRAIN_SPLIT" \
  --eval-split "$EVAL_SPLIT" \
  --all-langs "$ALL_LANGS" \
  --lang "$TARGET_LANG" \
  --seed "$SEED" \
  --resource-path "$TH_RESOURCE_PATH" \
  --detector-path "$TH_DETECTOR_PATH" \
  --byt5-model-path "$TH_BYT5_MODEL" \
  --submission-dir "$SUBMISSION_DIR" \
  --zip-path "$ZIP_PATH" \
  --debug-path "$DEBUG_PATH" \
  --use-detector "$USE_DETECTOR" \
  --use-byt5 "$USE_BYT5" \
  --use-rules "$USE_RULES" \
  --use-errdict "$USE_ERRDICT" \
  --use-mfr-candidate "$USE_MFR_CANDIDATE" \
  --detector-threshold "$DETECTOR_THRESHOLD" \
  --byt5-threshold "$BYT5_THRESHOLD" \
  --min-mfr-conf "$MIN_MFR_CONF" \
  --force-mfr-conf "$FORCE_MFR_CONF" \
  --min-mfr-count "$MIN_MFR_COUNT" \
  --accept-score-min "$ACCEPT_SCORE_MIN" \
  --max-edit-ratio "$MAX_EDIT_RATIO" \
  --byt5-bonus "$BYT5_BONUS" \
  --require-correct-dict-for-byt5 "$REQUIRE_CORRECT_DICT_FOR_BYT5" \
  --protect-nonthai "$PROTECT_NONTHAI" \
  --max-rule-candidates "$MAX_RULE_CANDIDATES" \
  --max-errdict-candidates "$MAX_ERRDICT_CANDIDATES" \
  --input-format "$INPUT_FORMAT" \
  --max-input-length "$MAX_INPUT_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-beams "$NUM_BEAMS" \
  --num-return-sequences "$NUM_RETURN_SEQUENCES" \
  --batch-size "$INFER_BATCH_SIZE" \
  --print-examples "$PRINT_EXAMPLES" \
  --print-lang-metrics "$PRINT_LANG_METRICS"

echo "============================================================"
echo "DONE"
echo "============================================================"
echo "zip:   $ZIP_PATH"
echo "debug: $DEBUG_PATH"
