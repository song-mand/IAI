#!/usr/bin/env bash
set -euo pipefail

# Make this script runnable from either project root or scripts/hr.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Submission policy:
#   HR: RandomForest gate -> ByT5 only for changed-token candidates
#   non-HR: MFR
python "$SCRIPT_DIR/sub_hr_rf_byt5.py" "$@"
