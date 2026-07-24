#!/usr/bin/env bash
# Syncs figures + generated macros into paper/ and compiles dr-pinns.tex.
# Fails (exit 2) if any [TBD:] placeholder survives in the PDF, i.e. if the
# text/figure/number synchronization is incomplete -- suitable for CI.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p paper/figures paper/generated
PAPER_FIGS=(control_set loss_single trajectory_vs_tube scalability
            ellipse_velocity_tube ellipse_selector_level ellipse_state_trajectory
            reference_extinction_curves training_history_relay
            dr_pinn_vs_reference branch_selection_diagnostic
            extinction_time_sweep
            residual_distribution threshold_sensitivity solver_convergence)
missing=0
for f in "${PAPER_FIGS[@]}"; do
    if [[ -f "results/figures/${f}.png" ]]; then
        cp "results/figures/${f}.png" paper/figures/
    else
        echo "!! missing figure: results/figures/${f}.png" >&2
        missing=1
    fi
done
for t in 61 62 63; do
    if [[ -f "results/aggregated/results_${t}.tex" ]]; then
        cp "results/aggregated/results_${t}.tex" paper/generated/
    else
        echo "!! missing macros: results/aggregated/results_${t}.tex" >&2
    fi
done
for f in results/aggregated/table_*.tex; do
    if [[ -f "$f" ]]; then cp "$f" paper/generated/; fi
done
if [[ $missing -eq 1 ]]; then
    echo ">>> (paper will compile with fallbacks/TBD)"
fi

if ! command -v pdflatex > /dev/null 2>&1; then
    echo ">>> pdflatex not available -- skipping compilation." >&2
    exit 0
fi
cd paper
pdflatex -interaction=nonstopmode dr-pinns.tex > /dev/null || true
if command -v bibtex > /dev/null 2>&1; then
    bibtex dr-pinns > /dev/null || true
fi
pdflatex -interaction=nonstopmode dr-pinns.tex > /dev/null || true
pdflatex -interaction=nonstopmode dr-pinns.tex > /dev/null

if command -v pdftotext > /dev/null 2>&1; then
    if pdftotext dr-pinns.pdf - 2>/dev/null | grep -q "\[TBD:"; then
        echo "!!! [TBD:] markers left in the PDF -- incomplete" \
             "synchronization." >&2
        exit 2
    fi
    echo ">>> PDF clean of [TBD:] markers -- full synchronization."
fi
