#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

# Make monoise_applied importable as a package.
touch ./monoise_applied/__init__.py

# ==============================
# Path settings
# ==============================

DATA_FILE="./data/train-00000-of-00001.parquet"
MAKE_HOLDOUT_SCRIPT="./scripts/make_es_it_holdout.py"

BASE_MODULE="monoise_applied.it_monoise_extend_candidates"
ACCEPT_MODULE="monoise_applied.it_monoise_accept_override"

# ==============================
# Split settings
# ==============================

LANGS="it"
TEST_SIZE=0.2
SEED=14

SPLIT_DIR="./eval_splits_it_accept_override"
SPLIT_PREFIX="it"

TRAIN_FILE="${SPLIT_DIR}/it_train.parquet"
VALID_FILE="${SPLIT_DIR}/it_valid.parquet"

# ==============================
# Model / report paths
# ==============================

BASE_MODEL_DIR="./models_it_monoise_extend_candidates_accept"
BASE_MODEL="${BASE_MODEL_DIR}/it_monoise_extend_candidates.joblib"

ACCEPT_MODEL_DIR="./models_it_monoise_accept_override"
ACCEPT_MODEL="${ACCEPT_MODEL_DIR}/it_accept_override.joblib"

REPORT_DIR="./reports"
REPORT_PATH="${REPORT_DIR}/it_accept_override_eval.txt"

# ==============================
# Candidate generation hyperparameters
# ==============================

# Max candidates from each candidate source.
# Higher value = higher candidate recall, but harder ranking.
TOP_K_PER_SOURCE=8

# Max number of split candidates.
# Higher value = more split candidates, but more noise.
MAX_SPLIT_CANDIDATES=8

# ==============================
# Base candidate ranker hyperparameters
# ==============================

BASE_N_ESTIMATORS=500
BASE_MAX_DEPTH=16
BASE_MIN_SAMPLES_LEAF=1
BASE_MIN_SAMPLES_SPLIT=2
BASE_MAX_FEATURES="sqrt"
BASE_CLASS_WEIGHT="balanced_subsample"

# ==============================
# Accept / override classifier hyperparameters
# ==============================

# Minimum probability required to accept an override.
# Higher value = fewer over-changes, but fewer corrections.
# Lower value = more corrections, but more over-changes.
ACCEPT_THRESHOLD=0.85

# Weight for bad changes on tokens that should stay unchanged.
# Higher value = more conservative.
# Lower value = more aggressive.
OVERCHANGE_NEGATIVE_WEIGHT=17.0

# Weight for good override examples.
# Higher value = more aggressive correction.
# Lower value = more conservative.
POSITIVE_WEIGHT=2.0

# Maximum depth of each RandomForest tree.
# Higher value = learns more complex patterns, but may overfit.
# Lower value = simpler and more conservative.
ACCEPT_MAX_DEPTH=10

# Minimum number of samples required in each leaf node.
# Higher value = more conservative and less overfitting.
# Lower value = more detailed learning.
ACCEPT_MIN_SAMPLES_LEAF=3

ACCEPT_N_ESTIMATORS=500
ACCEPT_MIN_SAMPLES_SPLIT=4
ACCEPT_MAX_FEATURES="sqrt"
ACCEPT_CLASS_WEIGHT="balanced_subsample"

# ==============================
# Helper: run module by import, not as __main__ file
# This prevents joblib pickle from saving custom classes as __main__.ClassName.
# ==============================

run_module() {
  local module_name="$1"
  shift

  python - "$module_name" "$@" <<'PY'
import importlib
import sys

module_name = sys.argv[1]
sys.argv = [module_name] + sys.argv[2:]

mod = importlib.import_module(module_name)
mod.main()
PY
}

echo "=============================="
echo "0. Check files and packages"
echo "=============================="

python - <<'PY'
import sklearn, joblib
print("scikit-learn/joblib OK")
PY

if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: DATA_FILE not found: $DATA_FILE"
  exit 1
fi

if [ ! -f "$MAKE_HOLDOUT_SCRIPT" ]; then
  echo "ERROR: MAKE_HOLDOUT_SCRIPT not found: $MAKE_HOLDOUT_SCRIPT"
  exit 1
fi

if [ ! -f "./monoise_applied/it_monoise_extend_candidates.py" ]; then
  echo "ERROR: base module file not found: ./monoise_applied/it_monoise_extend_candidates.py"
  exit 1
fi

if [ ! -f "./monoise_applied/it_monoise_accept_override.py" ]; then
  echo "ERROR: accept module file not found: ./monoise_applied/it_monoise_accept_override.py"
  exit 1
fi

echo "=============================="
echo "1. Remove previous accept override experiment files"
echo "=============================="

rm -rf "$SPLIT_DIR"
rm -rf "$BASE_MODEL_DIR"
rm -rf "$ACCEPT_MODEL_DIR"

mkdir -p "$BASE_MODEL_DIR"
mkdir -p "$ACCEPT_MODEL_DIR"
mkdir -p "$REPORT_DIR"

echo "=============================="
echo "2. Create IT holdout split"
echo "=============================="

python "$MAKE_HOLDOUT_SCRIPT" \
  --source_train_file "$DATA_FILE" \
  --langs "$LANGS" \
  --test_size "$TEST_SIZE" \
  --seed "$SEED" \
  --out_dir "$SPLIT_DIR" \
  --prefix "$SPLIT_PREFIX"

echo "=============================="
echo "3. Train base MoNoise extend-candidates ranker"
echo "=============================="

run_module "$BASE_MODULE" train \
  --train_file "$TRAIN_FILE" \
  --output "$BASE_MODEL" \
  --top_k_per_source "$TOP_K_PER_SOURCE" \
  --max_split_candidates "$MAX_SPLIT_CANDIDATES" \
  --n_estimators "$BASE_N_ESTIMATORS" \
  --max_depth "$BASE_MAX_DEPTH" \
  --min_samples_leaf "$BASE_MIN_SAMPLES_LEAF" \
  --min_samples_split "$BASE_MIN_SAMPLES_SPLIT" \
  --max_features "$BASE_MAX_FEATURES" \
  --class_weight "$BASE_CLASS_WEIGHT" \
  --seed "$SEED"

echo "=============================="
echo "4. Train accept override model"
echo "=============================="

run_module "$ACCEPT_MODULE" train \
  --train_file "$TRAIN_FILE" \
  --base_model "$BASE_MODEL" \
  --output "$ACCEPT_MODEL" \
  --n_estimators "$ACCEPT_N_ESTIMATORS" \
  --max_depth "$ACCEPT_MAX_DEPTH" \
  --min_samples_leaf "$ACCEPT_MIN_SAMPLES_LEAF" \
  --min_samples_split "$ACCEPT_MIN_SAMPLES_SPLIT" \
  --max_features "$ACCEPT_MAX_FEATURES" \
  --class_weight "$ACCEPT_CLASS_WEIGHT" \
  --positive_weight "$POSITIVE_WEIGHT" \
  --overchange_negative_weight "$OVERCHANGE_NEGATIVE_WEIGHT" \
  --seed "$SEED"

echo "=============================="
echo "5. Evaluate accept override model"
echo "=============================="

run_module "$ACCEPT_MODULE" eval \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --model "$ACCEPT_MODEL" \
  --accept_threshold "$ACCEPT_THRESHOLD" \
  --verbose | tee "$REPORT_PATH"

echo "=============================="
echo "Done."
echo "Report:"
echo "  $REPORT_PATH"
echo "=============================="

#   only threshold ablation
:<<'END'
python -m monoise_applied.it_monoise_accept_override eval \
   --train_file ./eval_splits_it_accept_override/it_train.parquet \
   --valid_file ./eval_splits_it_accept_override/it_valid.parquet \
   --model ./models_it_monoise_accept_override/it_accept_override.joblib \
   --accept_threshold 0.90 \
   --verbose | tee ./reports/it_accept_override_eval_t090.txt
END