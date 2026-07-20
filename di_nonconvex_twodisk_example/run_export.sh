#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_export.sh
#   ./run_export.sh outputs 3
#
# Arguments:
#   $1 output directory, default: outputs
#   $2 export scale, default: 3

OUT_DIR="${1:-outputs}"
SCALE="${2:-3}"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/generate_twodisk_figures.py --out "$OUT_DIR" --scale "$SCALE" --formats png,pdf,html

echo "Done. Files written to: $OUT_DIR"
