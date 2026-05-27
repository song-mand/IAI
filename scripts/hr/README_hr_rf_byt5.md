# HR RandomForest-gated ByT5 system

## Goal

Submission behavior is fixed as follows:

- `hr`: RandomForest classifier first predicts whether each token should change. Only predicted-changed tokens are normalized with the HR ByT5 model.
- non-`hr`: MFR only.

The submission script iterates over the evaluation split in the original dataset order. This avoids `raw` order mismatch errors in scorers that compare `label['raw']` and `pred['raw']` directly.

## Files

```text
scripts/hr/train_hr_rf_byt5.py
scripts/hr/sub_hr_rf_byt5.py
scripts/hr/run_hr_train.sh
scripts/hr/run_hr_sub.sh
scripts/hr/run_hr_full.sh
```

## Train

```bash
bash scripts/hr/run_hr_train.sh
```

Main outputs:

```text
final_model/hr_change_rf.joblib
final_model/hr_model/
```

Useful options:

```bash
# Train only RF gate
bash scripts/hr/run_hr_train.sh --skip_byt5

# Train only ByT5, assuming RF already exists
bash scripts/hr/run_hr_train.sh --skip_rf

# More conservative RF gate: tune threshold for precision instead of F1
bash scripts/hr/run_hr_train.sh --rf_threshold_metric precision

# Use only actually-changed tokens for ByT5 fine-tuning
bash scripts/hr/run_hr_train.sh --byt5_changed_only
```


## Change RandomForest parameters from bash

`run_hr_train.sh` exposes the RandomForest settings as bash environment variables.
You can either edit the default values inside the shell script or override them from the command line.

Example:

```bash
RF_N_ESTIMATORS=800 \
RF_MAX_DEPTH=30 \
RF_MIN_SAMPLES_LEAF=1 \
RF_MAX_FEATURES=sqrt \
RF_THRESHOLD_METRIC=precision \
bash scripts/hr/run_hr_train.sh
```

Supported RF variables:

```text
RF_N_ESTIMATORS        default 500
RF_CRITERION           gini | entropy | log_loss, default gini
RF_MAX_DEPTH           none or int, default none
RF_MIN_SAMPLES_SPLIT   default 2
RF_MIN_SAMPLES_LEAF    default 2
RF_MAX_FEATURES        none | sqrt | log2 | float ratio | int, default none
RF_MAX_LEAF_NODES      none or int, default none
RF_BOOTSTRAP           1 or 0, default 1
RF_MAX_SAMPLES         none | float ratio | int, default none
RF_CLASS_WEIGHT        none | balanced | balanced_subsample, default balanced_subsample
RF_N_JOBS              default -1
RF_VERBOSE             default 0
RF_VAL_RATIO           default 0.10
RF_THRESHOLD           default 0.50; used only when no validation tuning is done
RF_THRESHOLD_METRIC    f1 | precision, default f1
REFIT_RF_FULL          1 or 0, default 1
```

You can still pass Python arguments after the bash command. Arguments passed after the command override duplicated values because they appear later in the command line.

```bash
bash scripts/hr/run_hr_train.sh --rf_n_estimators 1000 --rf_max_depth 40
```

## Make submission

```bash
bash scripts/hr/run_hr_sub.sh
```

Output:

```text
submission.zip
submission_files/predictions.json
```

Override HR threshold at submission time:

```bash
bash scripts/hr/run_hr_sub.sh --hr_threshold 0.60
```

Fallback test: force every language including HR to use MFR:

```bash
bash scripts/hr/run_hr_sub.sh --force_mfr_hr
```

## Dependencies

Required packages include:

```bash
pip install datasets transformers peft torch scikit-learn joblib tqdm
```

## Path-safe bash execution

The run scripts are path-safe. You can run them either from the project root:

```bash
bash scripts/hr/run_hr_sub.sh
```

or from inside `scripts/hr`:

```bash
cd scripts/hr
bash run_hr_sub.sh
```

Outputs are still written relative to the project root, for example `submission.zip`, `submission_files/`, and `final_model/`.
