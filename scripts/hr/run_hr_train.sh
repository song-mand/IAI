#!/usr/bin/env bash
set -euo pipefail

# Make this script runnable from either project root or scripts/hr.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# HR two-stage training:
#   final_model/hr_change_rf.joblib
#   final_model/hr_model/
#
# RandomForest parameters can be changed here or overridden at runtime:
#   RF_N_ESTIMATORS=800 RF_MAX_DEPTH=30 bash scripts/hr/run_hr_train.sh
#   cd scripts/hr && RF_N_ESTIMATORS=800 RF_MAX_DEPTH=30 bash run_hr_train.sh
#
# Use "none" for sklearn None values.

RF_N_ESTIMATORS=${RF_N_ESTIMATORS:-500}
RF_CRITERION=${RF_CRITERION:-gini}                 # gini | entropy | log_loss
RF_MAX_DEPTH=${RF_MAX_DEPTH:-none}                # none or int
RF_MIN_SAMPLES_SPLIT=${RF_MIN_SAMPLES_SPLIT:-2}
RF_MIN_SAMPLES_LEAF=${RF_MIN_SAMPLES_LEAF:-2}
RF_MAX_FEATURES=${RF_MAX_FEATURES:-none}          # none | sqrt | log2 | float ratio | int
RF_MAX_LEAF_NODES=${RF_MAX_LEAF_NODES:-none}      # none or int
RF_BOOTSTRAP=${RF_BOOTSTRAP:-1}                   # 1 or 0
RF_MAX_SAMPLES=${RF_MAX_SAMPLES:-none}            # none | float ratio | int, only when bootstrap=1
RF_CLASS_WEIGHT=${RF_CLASS_WEIGHT:-balanced_subsample}  # none | balanced | balanced_subsample
RF_N_JOBS=${RF_N_JOBS:--1}
RF_VERBOSE=${RF_VERBOSE:-0}
RF_VAL_RATIO=${RF_VAL_RATIO:-0.10}
RF_THRESHOLD=${RF_THRESHOLD:-0.50}
RF_THRESHOLD_METRIC=${RF_THRESHOLD_METRIC:-f1}    # f1 | precision
REFIT_RF_FULL=${REFIT_RF_FULL:-1}

python "$SCRIPT_DIR/train_hr_rf_byt5.py" \
  --rf_n_estimators "$RF_N_ESTIMATORS" \
  --rf_criterion "$RF_CRITERION" \
  --rf_max_depth "$RF_MAX_DEPTH" \
  --rf_min_samples_split "$RF_MIN_SAMPLES_SPLIT" \
  --rf_min_samples_leaf "$RF_MIN_SAMPLES_LEAF" \
  --rf_max_features "$RF_MAX_FEATURES" \
  --rf_max_leaf_nodes "$RF_MAX_LEAF_NODES" \
  --rf_bootstrap "$RF_BOOTSTRAP" \
  --rf_max_samples "$RF_MAX_SAMPLES" \
  --rf_class_weight "$RF_CLASS_WEIGHT" \
  --rf_n_jobs "$RF_N_JOBS" \
  --rf_verbose "$RF_VERBOSE" \
  --rf_val_ratio "$RF_VAL_RATIO" \
  --rf_threshold "$RF_THRESHOLD" \
  --rf_threshold_metric "$RF_THRESHOLD_METRIC" \
  --refit_rf_full "$REFIT_RF_FULL" \
  "$@"
