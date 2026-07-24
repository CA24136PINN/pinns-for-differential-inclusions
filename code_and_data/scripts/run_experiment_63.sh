#!/usr/bin/env bash
# Example 6.3 (relay parabolic inclusion): TRAINING STAGE ONLY.
# Full run: 4 trainings x 40k epochs (GPU recommended; ~6 h wall clock).
# Writes results/raw/exp63/{raw_data.npz,manifest_63.json,checkpoints/}.
# Usage: bash scripts/run_experiment_63.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/run_experiment_63.py --stage train "$@"
