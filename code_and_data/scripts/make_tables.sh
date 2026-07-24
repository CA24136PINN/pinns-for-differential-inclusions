#!/usr/bin/env bash
# Regenerates results/aggregated/results_6X.tex (LaTeX macros) and the
# ready-made tabulars table_63_*.tex from the raw manifests. Cheap
# (seconds); requires the corresponding training/analysis stage.
set -euo pipefail
cd "$(dirname "$0")/.."
for exp in 61 62 63; do
    if [[ -f "results/raw/exp${exp}/manifest_${exp}.json" ]]; then
        python3 "experiments/run_experiment_${exp}.py" --stage tables
    else
        echo "!! results/raw/exp${exp}: no manifest -- skipping" \
             "(run scripts/run_experiment_${exp}.sh first)" >&2
    fi
done
