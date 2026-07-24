#!/usr/bin/env bash
# Example 6.3 (relay parabolic inclusion) -- referee-revision pipeline:
#   1. reference solver + convergence study        (CPU, minutes)
#   2. pre-registered training grid                (GPU, ~11 x 6 h)
#      3 seeds x lambda in {0.4,0.6,0.8} (adaptive) + ablation for
#      lambda=0.6: uniform sampling and residual-RAR (same seed).
#      Jobs are distributed DYNAMICALLY over free GPUs by
#      scripts/gpu_launcher.py (shared machine: only cards with >= 8 GB
#      free are used; resumable -- rerun this script after interruption).
#   3. analysis from checkpoints (thresholds, residual stats, aggregation)
# Figures/tables are generated separately (make_figures.sh / make_tables.sh).
# Usage: bash scripts/run_experiment_63.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 experiments/run_experiment_63.py --stage reference "$@"
python3 scripts/gpu_launcher.py "$@"
python3 experiments/run_experiment_63.py --stage analyze "$@"
