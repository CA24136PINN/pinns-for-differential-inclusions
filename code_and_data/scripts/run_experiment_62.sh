#!/usr/bin/env bash
# Example 6.2 (rotating ellipse + companion DR-PINN): TRAINING STAGE ONLY.
# Writes results/raw/exp62/{raw_data.npz,manifest_62.json,checkpoints/}.
# Usage: bash scripts/run_experiment_62.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/run_experiment_62.py --stage train "$@"
