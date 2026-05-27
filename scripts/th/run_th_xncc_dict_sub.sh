#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Thai XNCC-style extendable dictionary submission
# - TH: strict error_dict -> correct_dict only
# - non-TH: MFR baseline
# ============================================================

export DATASET=${DATASET:-"weerayut/multilexnorm2026-dev-pub"}
export TRAIN_SPLIT=${TRAIN_SPLIT:-"train"}
export EVAL_SPLIT=${EVAL_SPLIT:-"test"}
export ALL_LANGS=${ALL_LANGS:-"en,da,de,es,hr,it,nl,sl,sr,tr,iden,trde,id,ja,ko,th,vi"}
export TARGET_LANG=${TARGET_LANG:-"th"}

export TH_RESOURCE_PATH=${TH_RESOURCE_PATH:-"models/th/th_resources.pkl"}
export SUBMISSION_DIR=${SUBMISSION_DIR:-"submission_files"}
export ZIP_PATH=${ZIP_PATH:-"submission.zip"}
export DEBUG_PATH=${DEBUG_PATH:-"submission_files/th_xncc_dict_debug.json"}

# Strict dictionary evidence thresholds.
export XNCC_MIN_PAIR_COUNT=${XNCC_MIN_PAIR_COUNT:-1}
export XNCC_MIN_PAIR_CONF=${XNCC_MIN_PAIR_CONF:-0.40}
export XNCC_MIN_CHANGE_RATE=${XNCC_MIN_CHANGE_RATE:-0.40}
export XNCC_MIN_TOTAL_COUNT=${XNCC_MIN_TOTAL_COUNT:-3}

# Validity filters.
export XNCC_REQUIRE_CORRECT_DICT=${XNCC_REQUIRE_CORRECT_DICT:-1}
export XNCC_PROTECT_NONTHAI=${XNCC_PROTECT_NONTHAI:-1}
export XNCC_PROTECT_KNOWN_CORRECT=${XNCC_PROTECT_KNOWN_CORRECT:-1}
export XNCC_KNOWN_CORRECT_KEEP_RATE=${XNCC_KNOWN_CORRECT_KEEP_RATE:-0.15}
export XNCC_MAX_EDIT_RATIO=${XNCC_MAX_EDIT_RATIO:-0.75}
export XNCC_MAX_LEN_RATIO=${XNCC_MAX_LEN_RATIO:-1.80}
export XNCC_MAX_LEN_ADD=${XNCC_MAX_LEN_ADD:-3}
export XNCC_ALLOW_LONG_EXPANSION=${XNCC_ALLOW_LONG_EXPANSION:-0}
export XNCC_ALLOW_ABBREV_EXPANSION=${XNCC_ALLOW_ABBREV_EXPANSION:-0}

# Optional manual pair blacklist. Add risky pairs from debug analysis here.
export XNCC_BLOCK_PAIRS=${XNCC_BLOCK_PAIRS:-"ก่อ=>ก่อน,ค้า=>คะ,แอป=>แอปพลิเคชัน,เชี่ย=>เหี้ย,เห้=>เหี้ย"}

export PRINT_EXAMPLES=${PRINT_EXAMPLES:-120}
export PRINT_LANG_METRICS=${PRINT_LANG_METRICS:-1}

mkdir -p "$SUBMISSION_DIR"

echo "============================================================"
echo "Make Thai XNCC-dictionary submission"
echo "============================================================"
python sub_th_xncc_dict.py \
  --dataset "$DATASET" \
  --train-split "$TRAIN_SPLIT" \
  --eval-split "$EVAL_SPLIT" \
  --all-langs "$ALL_LANGS" \
  --lang "$TARGET_LANG" \
  --resource-path "$TH_RESOURCE_PATH" \
  --submission-dir "$SUBMISSION_DIR" \
  --zip-path "$ZIP_PATH" \
  --debug-path "$DEBUG_PATH" \
  --min-pair-count "$XNCC_MIN_PAIR_COUNT" \
  --min-pair-conf "$XNCC_MIN_PAIR_CONF" \
  --min-change-rate "$XNCC_MIN_CHANGE_RATE" \
  --min-total-count "$XNCC_MIN_TOTAL_COUNT" \
  --require-correct-dict "$XNCC_REQUIRE_CORRECT_DICT" \
  --protect-nonthai "$XNCC_PROTECT_NONTHAI" \
  --protect-known-correct "$XNCC_PROTECT_KNOWN_CORRECT" \
  --known-correct-keep-rate "$XNCC_KNOWN_CORRECT_KEEP_RATE" \
  --max-edit-ratio "$XNCC_MAX_EDIT_RATIO" \
  --max-len-ratio "$XNCC_MAX_LEN_RATIO" \
  --max-len-add "$XNCC_MAX_LEN_ADD" \
  --allow-long-expansion "$XNCC_ALLOW_LONG_EXPANSION" \
  --allow-abbrev-expansion "$XNCC_ALLOW_ABBREV_EXPANSION" \
  --block-pairs "$XNCC_BLOCK_PAIRS" \
  --print-examples "$PRINT_EXAMPLES" \
  --print-lang-metrics "$PRINT_LANG_METRICS"

echo "============================================================"
echo "DONE"
echo "============================================================"
echo "zip:   $ZIP_PATH"
echo "debug: $DEBUG_PATH"
