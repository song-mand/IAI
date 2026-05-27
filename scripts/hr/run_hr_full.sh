#!/usr/bin/env bash
set -euo pipefail

# Make this script runnable from either project root or scripts/hr.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Train with the same RandomForest environment variables supported by run_hr_train.sh.
bash "$SCRIPT_DIR/run_hr_train.sh" "$@"
python "$SCRIPT_DIR/sub_hr_rf_byt5.py"
