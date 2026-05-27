# Japanese JP-Copy-MFR-ContextByT5 Scheme

이 압축 파일은 **폴더 없이 파일만 바로 들어 있는 버전**입니다.
사용자는 이 파일들을 그대로 `scripts/jp/` 폴더 안에 넣으면 됩니다.

```text
scripts/jp/jp_scheme_common.py
scripts/jp/train_jp_scheme_byt5.py
scripts/jp/eval_jp_scheme.py
scripts/jp/sub_jp_scheme.py
scripts/jp/run_jp_train_scheme.sh
scripts/jp/run_jp_eval_scheme.sh
scripts/jp/run_jp_sub_scheme.sh
scripts/jp/README_jp_scheme.md
```

## Scheme

```text
COPY first
-> high-confidence MFR for stable raw->norm pairs
-> ByT5 only for context-sensitive/changeable Japanese tokens
-> safety filter for deletion, long outputs, and suspicious script jumps
```

## 경로 구조

이 스크립트들은 자신이 `scripts/jp/` 안에 있다고 가정합니다.
따라서 아래 두 방식 모두 동작합니다.

repo 루트에서 실행:

```bash
bash scripts/jp/run_jp_train_scheme.sh
bash scripts/jp/run_jp_eval_scheme.sh
bash scripts/jp/run_jp_sub_scheme.sh
```

또는 `scripts/jp`로 들어가서 실행:

```bash
cd scripts/jp
bash run_jp_train_scheme.sh
bash run_jp_eval_scheme.sh
bash run_jp_sub_scheme.sh
```

parquet 파일은 자동으로 아래 위치들을 순서대로 찾습니다.

```text
<repo>/
<repo>/data/
<repo>/datasets/
<repo>/input/
scripts/jp/
현재 실행 위치
```

파일명이 다음과 같으면 자동 탐색됩니다.

```text
train-00000-of-00001.parquet
validation-00000-of-00001.parquet
test-00000-of-00001.parquet
```

기본 출력 위치도 repo 루트 기준입니다.

```text
<repo>/final_model/jp_scheme_byt5
<repo>/final_model/jp_scheme_artifacts/jp_scheme_artifacts.json
<repo>/submission_files/predictions.json
<repo>/submission.zip
```

## 언어 코드와 parquet 파일 위치


업로드된 MultiLexNorm parquet에서 일본어 라벨은 `ja`입니다. 그래서 이 스크립트는 파일명과 폴더명은 `jp`를 쓰더라도, 내부 데이터 필터는 기본적으로 `JP_LANG_CODE=ja`를 사용합니다.

정말 데이터셋 안의 일본어 라벨이 `jp`라면 이렇게 실행합니다.

```bash
JP_LANG_CODE=jp bash scripts/jp/run_jp_train_scheme.sh
```

예를 들어 parquet가 자동 탐색 위치가 아닌 곳에 있으면 repo 루트에서 이렇게 실행하면 됩니다.

```bash
TRAIN_PARQUET=data/train-00000-of-00001.parquet \
VALID_PARQUET=data/validation-00000-of-00001.parquet \
TEST_PARQUET=data/test-00000-of-00001.parquet \
bash scripts/jp/run_jp_sub_scheme.sh
```

절대경로도 가능합니다.

```bash
TRAIN_PARQUET=/home/soohyun/iai_code/data/train-00000-of-00001.parquet \
VALID_PARQUET=/home/soohyun/iai_code/data/validation-00000-of-00001.parquet \
TEST_PARQUET=/home/soohyun/iai_code/data/test-00000-of-00001.parquet \
bash scripts/jp/run_jp_sub_scheme.sh
```

## Validation experiment

검증 실험에서는 validation을 학습에 넣지 않는 것이 맞습니다.

```bash
USE_VALIDATION_FOR_TRAINING=0 bash scripts/jp/run_jp_train_scheme.sh
bash scripts/jp/run_jp_eval_scheme.sh
```

ByT5 없이 copy + high-confidence MFR만 확인하려면:

```bash
NO_BYT5=1 bash scripts/jp/run_jp_eval_scheme.sh
```

## Final submission

최종 제출용에서는 validation 라벨까지 artifact/MFR 커버리지에 쓸 수 있습니다.

```bash
USE_VALIDATION_FOR_TRAINING=1 bash scripts/jp/run_jp_train_scheme.sh
INCLUDE_VALIDATION_FOR_MFR=1 REBUILD_ARTIFACTS=1 bash scripts/jp/run_jp_sub_scheme.sh
```

제출 스크립트는 test parquet의 원래 row 순서를 그대로 보존합니다.
일본어는 hybrid scheme을 사용하고, 일본어 외 언어는 plain MFR을 사용합니다.

## 주요 옵션

```bash
JP_EPOCHS=3
JP_LR=3e-5
JP_BATCH_SIZE=8
JP_LANG_CODE=ja
USE_LORA=0
JP_UNCHANGED_SAMPLE_RATE=0.08
JP_PUNCT_UNCHANGED_RATE=0.50
NO_BYT5=0
```

GPU 메모리가 부족하면 다음처럼 실행합니다.

```bash
USE_LORA=1 bash scripts/jp/run_jp_train_scheme.sh
```


## v3 note

`train_jp_scheme_byt5.py` now uses `processing_class=tokenizer` instead of the deprecated/removed `tokenizer=` argument for `Seq2SeqTrainer`.


## v4 note: loss=0 / grad_norm=nan

If training prints `loss: 0` and `grad_norm: nan`, stop the run and use the v4 defaults.
The most common cause is unstable fp16 full fine-tuning of ByT5/T5. This version disables fp16 by default and lowers the default learning rate to `3e-5`.

Recommended stable command:

```bash
JP_FP16=0 JP_LR=3e-5 JP_BATCH_SIZE=8 bash run_jp_train_scheme.sh
```

For a faster but safer low-memory run, use LoRA:

```bash
USE_LORA=1 JP_FP16=0 JP_LR=3e-4 bash run_jp_train_scheme.sh
```


## v5 note: Codabench raw-order assertion

Codabench scoring first checks that `label['raw']` and submitted `pred['raw']` are identical.
The official baseline submission code groups predictions by language, so v5 defaults to that language-grouped order.
For local experiments only, use `--preserve_row_order` directly with `sub_jp_scheme.py` if your local label file is in physical parquet row order.


## v6 note: official test source

`run_jp_sub_scheme.sh` now uses `load_dataset("weerayut/multilexnorm2026-dev-pub")["test"]` as the default prediction target, matching the IT baseline style. This avoids accidentally reading a local `test-00000-of-00001.parquet` with 11,956 rows.

For Codabench submission, run normally:

```bash
bash run_jp_sub_scheme.sh
```

Only for local experiments with a known-good parquet row order/count:

```bash
USE_LOCAL_TEST=1 TEST_PARQUET=/path/to/test-00000-of-00001.parquet bash run_jp_sub_scheme.sh
```
