# Revised Section 6.3 — dead-zone relay benchmark (Phi_eps)

Replaces the exact-relay parabolic experiment with the boundary-layer relay

    Phi_eps(s) = {-lam} (s>eps) | [-lam,lam] (|s|<=eps) | {lam} (s<-eps)

**Why.** The post-extinction set {u* = 0} has positive measure in Q; for a
smooth ansatz the exact-relay residual is bounded below by ~lam^2 there, so
the hypothesis J(u_n) -> 0 of Theorem 5.3 is unreachable by smooth candidates
(only chattering minimising sequences realise it). Phi_eps keeps (P1)-(P3)
verbatim, admits the continuous selection -lam*sat(s/eps) (so Theorem 2.12
applies too), contains the relay flow as an admissible element, and makes the
hypothesis of Theorem 5.3 smoothly realisable. Non-uniqueness returns: for
eps >= lam*||v1||_inf (torsion function v1), u = lam*v1 is an explicit second
solution — printed by stage 1 and encoded in \RelayWitnessThreshold.

## Layout

    configs/relay_eps.yaml            all knobs (sweeps, seeds, optimiser)
    src/dr_pinns/relay_eps/
      multifunction.py                dist^2 / projection / branches of Phi_eps
      model.py                        ansatz (6.26), operator via nested tapes
      sampling.py                     uniform | front (margin-fixed) | RAR
      training.py                     Adam hold-then-decay, J_hat curve
      reference.py                    splitting solver, torsion v1, refinement
      metrics.py                      dense residual (stratified), band times,
                                      branch cloud, witness distances
    experiments/relay_eps/            staged pipeline (1 reference, 2 train,
                                      3 eval, 4 figures, 5 macros)
    scripts/launch_relay_eps.sh       dynamic GPU queue (restart-safe)
    scripts/run_all_relay_eps.sh      end-to-end

## Run

    pip install tensorflow scipy pyyaml matplotlib
    scripts/run_all_relay_eps.sh "0" --smoke     # ~minutes, pipeline check
    scripts/run_all_relay_eps.sh "0 1 2 3"       # full campaign (45 runs)

Run matrix: eps-sweep {0.1,0.05,0.02,0.01,0} x 5 seeds at lam=0.6 (front);
lam-sweep {0.4,0.8} x 5 seeds at eps=0.05; sampler ablation
{uniform, RAR} x 5 seeds at the baseline. All outputs under
`results/relay_eps/` (add to the sync_update.sh whitelist); macros in
`generated/results_63eps.tex` with \providecommand fallbacks.

## Claim -> artefact map

| Paper claim (revised 6.3)                          | Artefact |
|----------------------------------------------------|----------|
| J_hat -> small for eps > 0 (Thm 5.3 hypothesis)     | fig_training.pdf, \RelayResMean* |
| plateau ~lam^2 as eps -> 0 (chattering remark)      | fig_eps_sweep.pdf |
| band-entry times monotone in lam                    | fig_band_times.pdf, \RelayTBandMonotone |
| cloud lies in graph of Phi_eps (0/N violations)     | fig_branch_cloud.pdf, \RelayBaselineViolations |
| hard pool tracks the front, not the boundary layer  | fig_hardpool.pdf (margin fix in sampling.py) |
| front sampling vs uniform vs RAR                    | fig_ablation.pdf, \RelayAblation* |
| residual strata pre/front/post                      | \RelayBaselineResMean{Pre,Front,Post} |
| pre-extinction agreement with relay reference       | \RelayBaselineNormDisc |
| non-uniqueness witness lam*v1                       | stage 1 stdout, tail_dist_to_witness in eval.json |
