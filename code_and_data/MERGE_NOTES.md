# Merge notes: relay_eps_package -> code_and_data (2026-07-29)

## Added (verbatim from relay_eps_package, no edits)
- src/dr_pinns/relay_eps/{__init__,model,multifunction,sampling,training,reference,metrics}.py
- experiments/relay_eps/{run_matrix,stage_reference,stage_train,stage_eval}.py
- experiments/relay_eps/README.md          (package README, incl. claim->artefact map)
- configs/relay_eps.yaml

## Added with edits
- experiments/relay_eps/stage_figures.py   graceful degradation: fig_training /
    fig_norms_* / fig_branch_cloud are SKIPPED with a warning (shipped PDFs kept)
    when per-run history.npz / eval_arrays.npz are absent, i.e. when only the
    shipped evaluation artifacts are present. Full behaviour unchanged when
    training artifacts exist (verified end-to-end).
- experiments/relay_eps/stage_macros.py    default --out changed:
    generated/results_63eps.tex -> results/aggregated/results_63eps.tex
    (main-repo convention; build_paper.sh syncs into paper/generated/).
- scripts/launch_relay_eps.sh              BUG FIX: --config from the extra args
    is now forwarded to run_matrix.py --list; previously the queue always
    enumerated the default 45-run matrix regardless of the config passed to
    stage_train ("unknown run id" failures). Also python -> python3.
- scripts/run_all_relay_eps.sh             accepts `--smoke` as first arg
    (GPU ids then via RELAY_EPS_GPUS, default "0"); python -> python3;
    final message updated to the new macro path.

## Modified in the main repo
- reproduce_all.sh        new step 3b after the 61/62/63 loop:
    bash scripts/run_all_relay_eps.sh "${RELAY_EPS_GPUS:-0}" $SMOKE
    (restart-safe: runs with existing weights.npz are skipped). Header updated.
- scripts/build_paper.sh  syncs results/relay_eps/figures/{fig_training,
    fig_band_times, fig_eps_sweep, fig_ablation, fig_hardpool, fig_branch_cloud,
    fig_norms_lam0.4/0.6/0.8}.pdf -> paper/figures/ and
    results/aggregated/results_63eps.tex -> paper/generated/ (loop: 61 62 63 63eps).
- requirements.txt, environment.yml        + pyyaml>=6.0
- README.md               new section "Example 6.3, revised"; layout tree updated.
- paper/dr-pinns.tex      replaced by the revised manuscript (eps-banded 6.3);
    previous version kept as paper/dr-pinns_pre_relay_eps.tex.bak.

## Shipped precomputed results (results/relay_eps/)
- runs/<45 ids>/{eval.json, run_config.json}, logs/, figures/*.pdf,
  eval_summary.json  -- from the full GPU campaign, untouched.
- reference/           refinement.csv from the campaign + ref_lam*.npz,
  torsion.npz regenerated here by stage_reference (CPU, minutes; refinement.csv
  regenerated bit-identical to the campaign's, 27/27 rows).
- NOT shipped (size): per-run weights.npz / history.npz / eval_arrays.npz /
  pool_history.npz. stage_figures+stage_macros work without them (see above);
  stage_eval and fig_training/fig_norms/fig_branch_cloud regeneration require
  them (they live with the local training outputs).
- The relay_eps_smoke results directory was intentionally NOT merged
  (pipeline-check artifacts only).

## Tested here
1. stage_reference (full): t* = 0.1845/0.1645/0.1505; refinement.csv identical
   to the GPU campaign's (27 rows).
2. stage_figures + stage_macros on shipped data: regenerates fig_eps_sweep,
   fig_band_times, fig_ablation + 31 macros; warns-and-keeps for the four
   figures needing training artifacts.
3. End-to-end on a scratch copy with a 4-run tiny config (CPU, --smoke):
   reference -> launch_relay_eps.sh queue (2 workers) -> eval -> figures ->
   macros, all exit 0; restart-safety confirmed (second launch skips all).
4. scripts/build_paper.sh: syncs 9 relay PDFs + results_63eps.tex, compiles the
   revised dr-pinns.tex, "[TBD:]" check passes.
