#!/usr/bin/env bash
# Kopiuje figury z results/ do figures/ (katalogu uzywanego przez paper.tex).
set -e
cd "$(dirname "$0")/.."
mkdir -p figures
cp results/results_linear_control_qp_quickhull/{control_set,loss_single,trajectory_vs_tube,scalability}.png figures/
cp results/results_rotating_ellipse/{ellipse_velocity_tube,ellipse_selector_level,ellipse_state_trajectory}.png figures/
cp results/results_parabolic_case/{reference_extinction_curves,training_history_relay,dr_pinn_vs_reference,branch_selection_diagnostic,extinction_time_sweep}.png figures/
echo "figures/ zsynchronizowane (12 figur papera)."
