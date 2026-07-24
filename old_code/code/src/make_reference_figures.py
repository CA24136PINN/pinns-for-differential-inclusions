"""Regenerate the training-free reference figure of Section 6.3.

Produces ``reference_extinction_curves.png`` (Figure ``fig:relay-extinction-ref``)
in the unified paper style, WITHOUT any network training, and verifies the
reference extinction times against the values reported in the manuscript
(Table ``tab:relay-extinction``).

The solver here is a verbatim copy of the operator-splitting scheme used in
``dr_pinn_relay_parabolic_experiment.ipynb`` (implicit 5-point diffusion +
exact soft-thresholding proximal step), so the figure is bit-identical in
content to the notebook's cell; only the export style differs (Computer
Modern 14pt, dpi=180 via paper_style.apply_paper_style).

Run from this directory:

    python make_reference_figures.py

Runtime: a few seconds on CPU. Exits nonzero if the recomputed extinction
times disagree with the manuscript table.
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import apply_paper_style

# ---------------------------------------------------------------------------
# Problem data (must match the notebook and Section 6.3 of the manuscript)
# ---------------------------------------------------------------------------
LAMBDA_SWEEP = [0.0, 0.4, 0.6, 0.8]  # 0.0 = classical heat-equation control case
T_HORIZON = 0.30
DT = 5e-4
N_GRID = 49

# Reference extinction times reported in Table `tab:relay-extinction`.
PAPER_TABLE_TSTAR = {0.4: 0.1845, 0.6: 0.1645, 0.8: 0.1505}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "results_parabolic_case")


def u0_numpy(X, Y):
    """Initial datum u0(x,y) = 16 x(1-x) y(1-y)."""
    return 16.0 * X * (1 - X) * Y * (1 - Y)


def build_laplacian(N):
    """Sparse 2D 5-point Laplacian, homogeneous Dirichlet BC, N x N interior grid."""
    h = 1.0 / (N + 1)
    e = np.ones(N)
    L1 = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1]) / h**2
    I1 = sp.identity(N)
    Lap = (sp.kron(L1, I1) + sp.kron(I1, L1)).tocsc()
    return Lap, h


def soft_threshold(v, tau):
    """Proximal operator of tau * |.| (elementwise)."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def run_reference_solver(lam, T, dt, N=N_GRID):
    """Implicit-diffusion + soft-threshold splitting; returns (times, l2norms, t*)."""
    Lap, h = build_laplacian(N)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")

    n_steps = int(round(T / dt))
    Id = sp.identity(N * N, format="csc")
    solve = spla.factorized((Id - dt * Lap).tocsc())

    u = u0_numpy(X, Y).flatten()
    times = [0.0]
    l2norms = [np.sqrt(np.sum(u**2) * h**2)]
    extinction_time = None
    for n in range(1, n_steps + 1):
        v = solve(u)
        u = soft_threshold(v, dt * lam)
        nrm = np.sqrt(np.sum(u**2) * h**2)
        times.append(n * dt)
        l2norms.append(nrm)
        if extinction_time is None and nrm == 0.0:
            extinction_time = n * dt
    return np.array(times), np.array(l2norms), extinction_time


def main():
    apply_paper_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    print(f"{'lambda':>8} | {'t*_recomputed':>14} | {'t*_paper_table':>15}")
    ok = True
    for lam in LAMBDA_SWEEP:
        times, norms, tstar = run_reference_solver(lam, T_HORIZON, DT)
        results[lam] = (times, norms, tstar)
        expected = PAPER_TABLE_TSTAR.get(lam)
        shown = f"{tstar:.4f}" if tstar is not None else "not extinguished"
        exp_shown = f"{expected:.4f}" if expected is not None else "(heat eq., n/a)"
        print(f"{lam:>8} | {shown:>14} | {exp_shown:>15}")
        if expected is not None and (tstar is None or abs(tstar - expected) > 1e-12):
            ok = False

    # Figure identical in content and style to the notebook cell that produces
    # `reference_extinction_curves.png` (global lambda->color map, y-axis
    # clipped at NORM_FLOOR, annotated extinction times).
    LAMBDA_COLORS = {0.0: "0.45", 0.4: "tab:blue", 0.6: "tab:orange", 0.8: "tab:green"}
    NORM_FLOOR = 1e-6
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for lam in LAMBDA_SWEEP:
        times, norms, tstar = results[lam]
        label = rf"$\lambda={lam}$" + (" (heat eq.)" if lam == 0.0 else "")
        ax.semilogy(times, np.maximum(norms, 1e-16), color=LAMBDA_COLORS[lam],
                    lw=2, label=label)
        if tstar is not None:
            ax.axvline(tstar, color=LAMBDA_COLORS[lam], linestyle=":",
                       linewidth=0.9, alpha=0.8)
            ax.annotate(rf"$t^*={tstar:.4f}$", xy=(tstar, NORM_FLOOR * 4),
                        xytext=(tstar + 0.004, NORM_FLOOR * 4), fontsize=10,
                        color=LAMBDA_COLORS[lam], rotation=90, va="bottom")
    ax.set_ylim(NORM_FLOOR, 2.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$  (log scale)")
    ax.set_title("Reference solution: finite-time extinction vs. classical decay")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "reference_extinction_curves.png")
    plt.savefig(out_path, dpi=180)
    print(f"\nSaved: {os.path.normpath(out_path)}")

    if not ok:
        print("ERROR: recomputed extinction times disagree with the manuscript "
              "table (tab:relay-extinction). Do NOT update the paper figure "
              "before resolving this.", file=sys.stderr)
        sys.exit(1)
    print("OK: all reference extinction times match Table tab:relay-extinction.")


if __name__ == "__main__":
    main()
