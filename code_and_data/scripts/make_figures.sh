#!/usr/bin/env bash
# Regenerates ALL paper figures into results/figures/ from the raw data
# (results/raw/exp6X/raw_data.npz). Cheap (seconds) -- no retraining:
# figures can be restyled/regenerated at will after a single training run.
set -euo pipefail
cd "$(dirname "$0")/.."
for exp in 61 62 63; do
    if [[ -f "results/raw/exp${exp}/raw_data.npz" ]]; then
        python3 "experiments/run_experiment_${exp}.py" --stage figures
    else
        echo "!! results/raw/exp${exp}: no raw_data.npz -- skipping" \
             "(run scripts/run_experiment_${exp}.sh first)" >&2
    fi
done
