#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Central reproduction entry point for the DR-PINN manuscript.
#
# Usage (from the repository root):
#
#   scripts/reproduce.sh refcurves   # Sec 6.3 reference figure only (CPU, seconds)
#   scripts/reproduce.sh exp1        # Sec 6.1 notebook (CPU ok, ~tens of minutes)
#   scripts/reproduce.sh exp2        # Sec 6.2 selector-steering experiment (CPU, minutes)
#   scripts/reproduce.sh exp2-illustrations  # Plotly geometry illustrations (optional)
#   scripts/reproduce.sh exp3        # Sec 6.3 notebook (GPU strongly recommended:
#                                    #   4 trainings x 40k epochs)
#   scripts/reproduce.sh sync        # copy freshly generated PNGs into paper/figures
#   scripts/reproduce.sh check       # audit paper/figures vs manuscript + style
#   scripts/reproduce.sh all         # refcurves + exp1 + exp2 + exp3 + sync + check
#
# Every experiment writes its figures into code/results/results_*/ (or
# replication_package/*/outputs/). Nothing touches paper/figures until you
# run the explicit `sync` step, so a failed run can never corrupt the
# manuscript figures.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# The 9 canonical manuscript figures and where each is generated.
declare -A FIGSRC=(
  [control_set.png]="code/results/results_linear_control_qp_quickhull"
  [loss_single.png]="code/results/results_linear_control_qp_quickhull"
  [trajectory_vs_tube.png]="code/results/results_linear_control_qp_quickhull"
  [scalability.png]="code/results/results_linear_control_qp_quickhull"
  [ellipse_velocity_tube.png]="code/results/results_rotating_ellipse"
  [ellipse_selector_level.png]="code/results/results_rotating_ellipse"
  [ellipse_state_trajectory.png]="code/results/results_rotating_ellipse"
  [reference_extinction_curves.png]="code/results/results_parabolic_case"
  [training_history_relay.png]="code/results/results_parabolic_case"
  [dr_pinn_vs_reference.png]="code/results/results_parabolic_case"
  [branch_selection_diagnostic.png]="code/results/results_parabolic_case"
  [extinction_time_sweep.png]="code/results/results_parabolic_case"
)

run_notebook () {
  local nb="$1"
  echo ">>> Executing notebook: $nb"
  ( cd code/src && jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 "$(basename "$nb")" )
}

do_refcurves () {
  echo ">>> [refcurves] Section 6.3 reference figure (no training)"
  ( cd code/src && python3 make_reference_figures.py )
}

do_exp1 () {
  echo ">>> [exp1] Section 6.1: linear control system (Quickhull + QP)"
  echo ">>> NOTE: regenerates loss/trajectory/scalability figures; per-epoch"
  echo ">>>       TIMINGS in scalability.png are hardware-dependent. If they"
  echo ">>>       change materially, the numbers quoted in Sec 6.1 must be"
  echo ">>>       updated to match -- do not sync silently."
  run_notebook code/src/dr_pinn_linear_control_qp_quickhull.ipynb
}

do_exp2 () {
  echo ">>> [exp2] Section 6.2: rotating-ellipse selector-steering experiment"
  echo ">>>        (72-variable L-BFGS-B continuation; CPU, ~2-4 minutes)"
  ( cd code/src && python3 rotating_ellipse_selector_experiment.py )
  echo ">>> [exp2] DR-PINN companion run (Newton-based ellipse projection;"
  echo ">>>        CPU, ~4-5 minutes; prints diagnostics quoted in Sec 6.2)"
  ( cd code/src && python3 rotating_ellipse_drpinn_companion.py )
}

do_exp2_illustrations () {
  echo ">>> [exp2-illustrations] Plotly geometry illustrations (NOT the paper"
  echo ">>>        experiment: hand-crafted selector, synthetic drift)"
  for d in replication_package/di_convex_ellipse_example \
           replication_package/di_nonconvex_twodisk_example; do
    ( cd "$d" && chmod +x run_export.sh && ./run_export.sh )
  done
}

do_exp3 () {
  echo ">>> [exp3] Section 6.3: parabolic relay inclusion"
  echo ">>> WARNING: 4 x 40000-epoch trainings; GPU strongly recommended."
  run_notebook code/src/dr_pinn_relay_parabolic_experiment.ipynb
}

do_sync () {
  echo ">>> [sync] Copying canonical figures into paper/figures/"
  for fig in "${!FIGSRC[@]}"; do
    src="${FIGSRC[$fig]}/$fig"
    if [[ -f "$src" ]]; then
      cp -v "$src" "paper/figures/$fig"
    else
      echo "    (skip) $src not present -- run the corresponding experiment first"
    fi
  done
}

do_check () {
  python3 scripts/check_figures.py
}

case "${1:-}" in
  refcurves) do_refcurves ;;
  exp1)      do_exp1 ;;
  exp2)      do_exp2 ;;
  exp2-illustrations) do_exp2_illustrations ;;
  exp3)      do_exp3 ;;
  sync)      do_sync ;;
  check)     do_check ;;
  all)       do_refcurves; do_exp1; do_exp2; do_exp3; do_sync; do_check ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,20p'; exit 1 ;;
esac
