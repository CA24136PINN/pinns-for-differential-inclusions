"""Stage 4 (CPU): all figures for the revised Section 6.3.

  fig_eps_sweep.pdf      achievable dense-grid residual vs eps (the headline
                         figure: continuous degradation to the ~lam^2 plateau
                         as eps -> 0, i.e. the explained boundary phenomenon)
  fig_training.pdf       J_hat validation curves vs epoch, one line per eps
                         (direct realisation of the hypothesis of Thm 5.3)
  fig_norms_lam{...}.pdf L2 decay: reference vs DR-PINN seeds, eps-band marked
  fig_band_times.pdf     band-entry times vs lam at eps = eps_baseline
                         (monotonicity restored, theory-native band)
  fig_branch_cloud.pdf   (s,z) cloud on the graph of Phi_eps + violation count
  fig_hardpool.pdf       spatial heat map + time marginal of hard-pool points
                         (confounder diagnostic: no boundary layer)
  fig_ablation.pdf       final residual: front vs uniform vs RAR (box per seed)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def load_summary(root: Path) -> list[dict]:
    return json.loads((root / "eval_summary.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relay_eps.yaml")
    ap.add_argument("--root", default="results/relay_eps")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.root)
    figdir = root / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    res = load_summary(root)
    lam0, eps0 = cfg["lam_baseline"], cfg["eps_baseline"]

    # ------------------------------------------------ fig_eps_sweep
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    eps_vals = sorted(cfg["eps_sweep"], reverse=True)
    for stat, marker, label in [("mean", "o", "mean"), ("max", "s", "max")]:
        med, lo, hi = [], [], []
        for e in eps_vals:
            vals = [r["residual"][stat] for r in res
                    if r["lam"] == lam0 and r["eps"] == e and r["sampler"] == "front"]
            vals = np.array(vals)
            med.append(np.median(vals)); lo.append(vals.min()); hi.append(vals.max())
        x = [max(e, 2e-3) for e in eps_vals]  # plot eps=0 at a pseudo-position
        ax.plot(x, med, marker=marker, label=f"dense-grid {label}")
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.axhline(lam0 ** 2, ls="--", c="k", lw=0.8,
               label=r"$\lambda^2$ plateau (smooth-ansatz bound)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$ (0 plotted at $2\times10^{-3}$)")
    ax.set_ylabel(r"dense-grid $\mathrm{dist}^2$ residual")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(figdir / "fig_eps_sweep.pdf"); plt.close(fig)

    # ------------------------------------------------ fig_training
    # Requires per-run history.npz (training artifact, not part of the
    # shipped evaluation results). Skip gracefully if absent, keeping the
    # shipped fig_training.pdf.
    hist_runs = [r for r in res if r["lam"] == lam0 and r["sampler"] == "front"]
    missing_hist = [r["run"] for r in hist_runs
                    if not (root / "runs" / r["run"] / "history.npz").exists()]
    if missing_hist:
        print(f"!! fig_training: {len(missing_hist)} run(s) without "
              "history.npz (training artifacts not shipped) -- keeping the "
              "shipped figure")
        skip_training = True
    else:
        skip_training = False
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(eps_vals):
        for r in res:
            if skip_training:
                break
            if r["lam"] == lam0 and r["eps"] == e and r["sampler"] == "front":
                h = np.load(root / "runs" / r["run"] / "history.npz")
                ax.plot(h["epochs"], h["val_mean"], c=cmap(i / max(len(eps_vals) - 1, 1)),
                        alpha=0.6, lw=0.8,
                        label=rf"$\varepsilon={e:g}$" if r["seed"] == cfg["seeds"][0] else None)
    if not skip_training:
        ax.set_yscale("log"); ax.set_xlabel("epoch")
        ax.set_ylabel(r"$\widehat{\mathcal{J}}(u_\theta)$ (validation)")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(figdir / "fig_training.pdf")
    plt.close(fig)

    # ------------------------------------------------ fig_norms per lam
    for lam in [lam0] + list(cfg["lam_sweep"]):
        ref = np.load(root / "reference" / f"ref_lam{lam:g}.npz")
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.plot(ref["times"], np.clip(ref["l2"], 1e-6, None), "k-", lw=1.5,
                label="reference (relay, exact)")
        eps_here = eps0
        n_curves = 0
        for r in res:
            if r["lam"] == lam and r["eps"] == eps_here and r["sampler"] == "front":
                ap_ = root / "runs" / r["run"] / "eval_arrays.npz"
                if not ap_.exists():
                    continue
                a = np.load(ap_)
                ax.plot(a["times"], np.clip(a["l2"], 1e-6, None), "--", lw=0.8,
                        alpha=0.8, label=f"DR-PINN s{r['seed']}")
                n_curves += 1
        ts = float(ref["t_star"])
        ax.axvline(ts, c="grey", ls=":", lw=0.8)
        ax.axhline(eps_here, c="tab:red", ls="-.", lw=0.8,
                   label=rf"band level $\varepsilon={eps_here:g}$")
        ax.set_yscale("log"); ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$")
        ax.set_title(rf"$\lambda={lam:g}$", fontsize=9)
        ax.legend(fontsize=6); fig.tight_layout()
        if n_curves > 0:
            fig.savefig(figdir / f"fig_norms_lam{lam:g}.pdf")
        else:
            print(f"!! fig_norms_lam{lam:g}: no run has eval_arrays.npz "
                  "(evaluation arrays not shipped) -- keeping the shipped "
                  "figure")
        plt.close(fig)

    # ------------------------------------------------ fig_band_times
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    lams = sorted([lam0] + list(cfg["lam_sweep"]))
    for lam in lams:
        ref = np.load(root / "reference" / f"ref_lam{lam:g}.npz")
        tb = [r["t_band"] for r in res
              if r["lam"] == lam and r["eps"] == eps0 and r["sampler"] == "front"
              and r["t_band"] is not None]
        ax.scatter([lam] * len(tb), tb, c="tab:blue", s=14, alpha=0.7,
                   label="DR-PINN band entry" if lam == lams[0] else None)
        ax.scatter([lam], [float(ref["t_star"])], marker="*", c="k", s=60,
                   label=r"reference $t^*$" if lam == lams[0] else None)
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$t_{\mathrm{band}}(\varepsilon)$")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(figdir / "fig_band_times.pdf"); plt.close(fig)

    # ------------------------------------------------ fig_branch_cloud
    pick = next(r for r in res if r["lam"] == lam0 and r["eps"] == eps0
                and r["sampler"] == "front" and r["seed"] == cfg["seeds"][0])
    cloud_path = root / "runs" / pick["run"] / "eval_arrays.npz"
    if cloud_path.exists():
        a = np.load(cloud_path)
        s, z = a["cloud_s"], a["cloud_z"]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.scatter(s, z, s=2, alpha=0.4, label="collocation cloud")
        e = eps0; lamv = lam0
        gs = np.linspace(min(s.min(), -2 * e), max(s.max(), 3 * e), 400)
        ax.plot(gs[gs > e], -lamv * np.ones((gs > e).sum()), "r-", lw=1.2,
                label=r"graph of $\Phi_\varepsilon$")
        ax.plot(gs[gs < -e], lamv * np.ones((gs < -e).sum()), "r-", lw=1.2)
        ax.add_patch(plt.Rectangle((-e, -lamv), 2 * e, 2 * lamv, fc="red",
                                   alpha=0.12, ec="r", lw=0.8))
        ax.set_xlabel(r"$s=u_\theta$")
        ax.set_ylabel(r"$z=\partial_t u_\theta-\Delta u_\theta$")
        ax.set_title(f"violations (dist$^2>${pick['violation_tol']:g}): "
                     f"{pick['violations']}/{pick['cloud_size']}", fontsize=8)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(figdir / "fig_branch_cloud.pdf"); plt.close(fig)
    else:
        print("!! fig_branch_cloud: eval_arrays.npz not shipped -- keeping "
              "the shipped figure")

    # ------------------------------------------------ fig_hardpool
    ph_path = root / "runs" / pick["run"] / "pool_history.npz"
    if ph_path.exists():
        ph = np.load(ph_path)
        allp = np.concatenate([ph[k] for k in ph.files], axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
        hb = axes[0].hexbin(allp[:, 1], allp[:, 2], gridsize=40, cmap="magma")
        axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$y$")
        axes[0].set_title("hard-pool spatial density", fontsize=8)
        fig.colorbar(hb, ax=axes[0])
        axes[1].hist(allp[:, 0], bins=60, color="tab:purple", alpha=0.8)
        axes[1].set_xlabel("$t$"); axes[1].set_title("hard-pool time marginal",
                                                     fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / "fig_hardpool.pdf"); plt.close(fig)

    # ------------------------------------------------ fig_ablation
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    modes = ["front"] + list(cfg["ablation_samplers"])
    data = []
    for m in modes:
        vals = [r["residual"]["mean"] for r in res
                if r["lam"] == lam0 and r["eps"] == eps0 and r["sampler"] == m]
        data.append(vals)
    ax.boxplot(data, tick_labels=modes)
    ax.set_yscale("log"); ax.set_ylabel("dense-grid mean residual")
    fig.tight_layout()
    fig.savefig(figdir / "fig_ablation.pdf"); plt.close(fig)

    print(f"figures -> {figdir}")


if __name__ == "__main__":
    main()
