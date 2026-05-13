#!/usr/bin/env bash
set -e

cd ~/iai_code
source .venv/bin/activate

echo "=============================="
echo "0. Check required packages"
echo "=============================="
python - <<'PY'
import sklearn, joblib
print("scikit-learn/joblib OK")
PY

# ==============================
# Path settings
# ==============================

# 원본 train parquet 파일
DATA_FILE="./data/train-00000-of-00001.parquet"

# holdout split 생성 코드
MAKE_HOLDOUT_SCRIPT="./scripts/make_es_it_holdout.py"

# 이번에 사용할 MoNoise-style 확장 candidate ranker 코드
MODEL_SCRIPT="./monoise_applied/it_monoise_extend_candidates.py"

# 경로 확인
if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: DATA_FILE not found: $DATA_FILE"
  exit 1
fi

if [ ! -f "$MAKE_HOLDOUT_SCRIPT" ]; then
  echo "ERROR: make_es_it_holdout.py not found: $MAKE_HOLDOUT_SCRIPT"
  exit 1
fi

if [ ! -f "$MODEL_SCRIPT" ]; then
  echo "ERROR: it_monoise_extend_candidates.py not found: $MODEL_SCRIPT"
  exit 1
fi

# ==============================
# Experiment-level settings
# ==============================

# LANGS:
# 실험할 언어입니다. 여기서는 Italian만 사용합니다.
LANGS="it"

# TEST_SIZE:
# train 중 validation으로 분리할 비율입니다.
# 0.2면 80% train / 20% valid입니다.
TEST_SIZE=0.2

# SEED:
# train/valid split 및 RandomForest 학습 재현성을 위한 seed입니다.
SEED=42

# SPLIT_DIR:
# 생성된 holdout split 저장 위치입니다.
SPLIT_DIR="./eval_splits_it_extend_candidates"

# SPLIT_PREFIX:
# 생성될 파일 prefix입니다.
# 결과: it_train.parquet, it_valid.parquet
SPLIT_PREFIX="it"

# MODEL_DIR:
# candidate ranker 모델 저장 폴더입니다.
MODEL_DIR="./models_it_monoise_extend_candidates"

# MODEL_PATH:
# 학습된 candidate ranker joblib 저장 경로입니다.
MODEL_PATH="${MODEL_DIR}/it_monoise_extend_candidates.joblib"

# REPORT_PATH:
# 평가 결과 저장 파일입니다.
REPORT_PATH="./reports/it_monoise_extend_candidates_eval.txt"

# ==============================
# Candidate generation hyperparameters
# ==============================

# TOP_K_PER_SOURCE:
# lookup/key/case/diacritic 등 각 candidate source에서 최대 몇 개 후보를 가져올지 결정합니다.
# 높이면 candidate upperbound가 올라갈 수 있지만 ranking이 어려워지고 over-normalization 위험도 커집니다.
TOP_K_PER_SOURCE=8

# MAX_SPLIT_CANDIDATES:
# split candidate를 최대 몇 개 만들지 결정합니다.
# 예: cé -> c' è 같은 후보를 만들 때 과도한 split 후보 생성을 막습니다.
MAX_SPLIT_CANDIDATES=8

# ==============================
# RandomForest ranker hyperparameters
# ==============================

# N_ESTIMATORS:
# RandomForest tree 개수입니다.
# 높이면 안정적이지만 학습/추론 시간이 늘어납니다.
N_ESTIMATORS=500

# MAX_DEPTH:
# tree 최대 깊이입니다.
# 높으면 복잡한 패턴을 학습하지만 과적합 위험이 증가합니다.
# 0이면 depth 제한 없음으로 처리됩니다.
MAX_DEPTH=16

# MIN_SAMPLES_LEAF:
# leaf node에 필요한 최소 샘플 수입니다.
# 낮으면 세밀하고 공격적으로 학습하고, 높이면 더 보수적으로 학습합니다.
MIN_SAMPLES_LEAF=1

# MIN_SAMPLES_SPLIT:
# node split에 필요한 최소 샘플 수입니다.
# 낮으면 더 세밀하게 split합니다.
MIN_SAMPLES_SPLIT=2

# MAX_FEATURES:
# 각 split에서 고려할 feature 수 방식입니다.
# sqrt는 RandomForest에서 안정적인 기본 선택입니다.
MAX_FEATURES="sqrt"

# CLASS_WEIGHT:
# gold candidate / wrong candidate imbalance 보정 방식입니다.
# balanced_subsample은 tree별 bootstrap sample 기준으로 class weight를 조정합니다.
CLASS_WEIGHT="balanced_subsample"

# ==============================
# Selection / decoding hyperparameters
# ==============================

# MARGIN:
# best candidate score가 original token score보다 최소 얼마나 높아야 normalize할지 결정합니다.
# 높이면 더 보수적으로 copy하고, 낮추면 더 공격적으로 normalize합니다.
MARGIN=0.10

# MIN_BEST_SCORE:
# best candidate의 최소 RF score입니다.
# 이 값보다 낮으면 best candidate라도 raw를 유지합니다.
MIN_BEST_SCORE=0.20

# ==============================
# Optional ByT5 candidate settings
# ==============================

# USE_BYT5_CANDIDATE:
# true면 ByT5 출력을 candidate source 중 하나로 추가합니다.
# IT에서는 Direct ByT5가 불안정했으므로 기본 false가 안전합니다.
USE_BYT5_CANDIDATE=false

# BYT5_MODEL_PATH:
# USE_BYT5_CANDIDATE=true일 때 사용할 ByT5 모델 경로입니다.
BYT5_MODEL_PATH="./final_model_eval_it/it_model"

# BYT5_MODE:
# missing이면 기존 candidate가 original밖에 없을 때만 ByT5 후보를 추가합니다.
# all이면 모든 non-protected token에 ByT5 후보를 추가하므로 IT에서는 위험할 수 있습니다.
BYT5_MODE="missing"

echo "=============================="
echo "1. Remove previous IT monoise_extend_candidates files"
echo "=============================="
rm -rf "$SPLIT_DIR"
rm -rf "$MODEL_DIR"

mkdir -p "$MODEL_DIR"
mkdir -p ./reports

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
echo "3. Train MoNoise extend-candidates IT candidate ranker"
echo "=============================="
python "$MODEL_SCRIPT" train \
  --train_file "${SPLIT_DIR}/it_train.parquet" \
  --output "$MODEL_PATH" \
  --top_k_per_source "$TOP_K_PER_SOURCE" \
  --max_split_candidates "$MAX_SPLIT_CANDIDATES" \
  --n_estimators "$N_ESTIMATORS" \
  --max_depth "$MAX_DEPTH" \
  --min_samples_leaf "$MIN_SAMPLES_LEAF" \
  --min_samples_split "$MIN_SAMPLES_SPLIT" \
  --max_features "$MAX_FEATURES" \
  --class_weight "$CLASS_WEIGHT" \
  --seed "$SEED"

echo "=============================="
echo "4. Evaluate MoNoise extend-candidates IT candidate ranker"
echo "=============================="

BYT5_ARGS=()
if [ "$USE_BYT5_CANDIDATE" = true ]; then
  if [ ! -d "$BYT5_MODEL_PATH" ]; then
    echo "ERROR: BYT5_MODEL_PATH not found: $BYT5_MODEL_PATH"
    exit 1
  fi

  BYT5_ARGS+=(--use_byt5_candidate)
  BYT5_ARGS+=(--byt5_model_path "$BYT5_MODEL_PATH")
  BYT5_ARGS+=(--byt5_mode "$BYT5_MODE")
fi

python "$MODEL_SCRIPT" eval \
  --train_file "${SPLIT_DIR}/it_train.parquet" \
  --valid_file "${SPLIT_DIR}/it_valid.parquet" \
  --model "$MODEL_PATH" \
  --margin "$MARGIN" \
  --min_best_score "$MIN_BEST_SCORE" \
  "${BYT5_ARGS[@]}" \
  --show_gold_candidates \
  --max_gold_candidate_examples 80 \
  --verbose | tee "$REPORT_PATH"

echo "=============================="
echo "Done."
echo "Report:"
echo "  $REPORT_PATH"
echo "=============================="