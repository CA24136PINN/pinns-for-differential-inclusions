#!/usr/bin/env bash
# =====================================================================
# DR-PINNs -- full replication in one command.
#
#   bash reproduce_all.sh            # full run (6.3 grid: ~66 GPU-h; ~3x6 h on 4 GPUs)
#   bash reproduce_all.sh --smoke    # end-to-end pipeline test (~5-10 min)
#
# Order:
#   0. scripts/system_info.sh          record hardware/software environment
#   1. scripts/run_experiment_61.sh    Example 6.1  (training stage only)
#   2. scripts/run_experiment_62.sh    Example 6.2  (training stage only)
#   3. scripts/run_experiment_63.sh    Example 6.3  (reference + GPU grid + analyze)
#   3b. scripts/run_all_relay_eps.sh   Example 6.3, revised (eps-banded relay):
#                                      reference -> 45-run GPU queue -> eval
#                                      -> figures -> macros; GPU ids via
#                                      RELAY_EPS_GPUS (default "0"); restart-
#                                      safe: runs with existing weights.npz
#                                      under results/relay_eps/runs/ are
#                                      skipped, so shipped results are reused.
#   4. scripts/make_tables.sh          LaTeX macros from the raw manifests
#   5. scripts/make_figures.sh         all figures from the raw data
#   6. scripts/build_paper.sh          sync into paper/ + pdflatex + TBD check
#
# Training (hours) is strictly separated from figure/table generation
# (seconds): steps 4-6 can be re-run at any time without touching 1-3.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

SMOKE="${1:-}"
if [[ -n "$SMOKE" && "$SMOKE" != "--smoke" ]]; then
    echo "Usage: $0 [--smoke]" >&2
    exit 1
fi
if [[ "$SMOKE" == "--smoke" ]]; then
    echo ">>> SMOKE MODE: truncated budgets, results not publication-grade."
fi

t0=$(date +%s)
bash scripts/system_info.sh
for exp in 61 62 63; do
    echo
    echo "=============================================================="
    echo ">>> Experiment 6.${exp:1:1} -- scripts/run_experiment_${exp}.sh $SMOKE"
    echo "=============================================================="
    bash "scripts/run_experiment_${exp}.sh" $SMOKE
done
echo
echo "=============================================================="
echo ">>> Experiment 6.3 (revised, eps-banded relay) --" \
     "scripts/run_all_relay_eps.sh"
echo "=============================================================="
bash scripts/run_all_relay_eps.sh "${RELAY_EPS_GPUS:-0}" $SMOKE
echo
bash scripts/make_tables.sh
bash scripts/make_figures.sh
bash scripts/build_paper.sh
echo
echo ">>> Done in $(( $(date +%s) - t0 )) s."
echo ">>> Raw data + manifests: results/raw/exp6{1,2,3}/"
echo ">>> Figures:              results/figures/"
echo ">>> Macros:               results/aggregated/"
