#!/usr/bin/env bash
# Example 6.1 (linear control + Quickhull): TRAINING STAGE ONLY.
# Writes results/raw/exp61/{raw_data.npz,manifest_61.json}.
# Figures/tables are generated separately (make_figures.sh / make_tables.sh).
# Usage: bash scripts/run_experiment_61.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/run_experiment_61.py --stage train "$@"
