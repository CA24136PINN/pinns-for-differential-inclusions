#!/usr/bin/env bash
# Full pipeline for the revised Section 6.3, in the staged order of the
# replication package. Training (stage 2) is the only expensive stage and is
# decoupled from figure/table generation, so stages 3-5 can be re-run cheaply.
#
# Usage:  scripts/run_all_relay_eps.sh "0 1 2 3"      # GPU ids
#         scripts/run_all_relay_eps.sh "0" --smoke    # end-to-end smoke test
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:experiments/relay_eps:${PYTHONPATH:-}"

# GPU ids: first positional arg, or RELAY_EPS_GPUS env (default "0").
# `run_all_relay_eps.sh --smoke` works too (GPUs then come from the env).
if [[ "${1:-}" == "--smoke" ]]; then
    GPUS="${RELAY_EPS_GPUS:-0}"
else
    GPUS="${1:-${RELAY_EPS_GPUS:-0}}"
    shift || true
fi

echo "=== stage 1: reference + torsion witness + refinement study ==="
python3 experiments/relay_eps/stage_reference.py "$@"

echo "=== stage 2: training (GPU queue) ==="
scripts/launch_relay_eps.sh "$GPUS" "$@"

echo "=== stage 3: evaluation ==="
python3 experiments/relay_eps/stage_eval.py

echo "=== stage 4: figures ==="
python3 experiments/relay_eps/stage_figures.py

echo "=== stage 5: LaTeX macros ==="
python3 experiments/relay_eps/stage_macros.py

echo "pipeline complete: results/relay_eps/ + results/aggregated/results_63eps.tex"
