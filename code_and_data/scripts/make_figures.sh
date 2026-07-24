#!/usr/bin/env bash
# Regenerates ALL paper figures into results/figures/ from raw/analysis
# data. Cheap (seconds) -- no retraining: figures can be restyled and
# regenerated at will after a single training run.
set -euo pipefail
cd "$(dirname "$0")/.."
for exp in 61 62; do
    if [[ -f "results/raw/exp${exp}/raw_data.npz" ]]; then
        python3 "experiments/run_experiment_${exp}.py" --stage figures
    else
        echo "!! results/raw/exp${exp}: no raw_data.npz -- skipping" >&2
    fi
done
if [[ -f "results/raw/exp63/analysis.npz" ]]; then
    python3 experiments/run_experiment_63.py --stage figures
else
    echo "!! results/raw/exp63: no analysis.npz -- skipping" \
         "(run scripts/run_experiment_63.sh first)" >&2
fi
