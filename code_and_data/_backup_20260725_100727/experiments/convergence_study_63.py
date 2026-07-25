"""Convergence study for the Section 6.3 reference solver.

Reproduces the first-order operator-splitting scheme for
    du/dt - Lap u  in  -lambda * Sign(u),   u|_{dOmega}=0,
    u0(x,y) = 16 x(1-x) y(1-y),  Omega=(0,1)^2,
and studies the stability of the detected extinction time t* under
(i) time-step refinement and (ii) mesh refinement.

Scheme (as in the paper):
  step 1: implicit diffusion   v = (I - dt*Lap_h)^{-1} u^n
  step 2: exact prox (soft-threshold)  u^{n+1} = sign(v)*max(|v|-lambda*dt, 0)

Extinction detection: first time at which the discrete field is exactly zero
(the soft-threshold can produce exact zeros), cross-checked with a
persistence check to the horizon.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time, json


def laplacian_2d(n, h):
    """5-point Laplacian on the n x n interior grid, Dirichlet BC."""
    e = np.ones(n)
    T = sp.diags([e[:-1], -2.0 * e, e[:-1]], [-1, 0, 1]) / h**2
    I = sp.identity(n)
    return (sp.kron(I, T) + sp.kron(T, I)).tocsc()


def run_reference(lam, n, dt, T=0.3):
    """Return (t_extinct, t_persistent, norm_history_times, norm_history)."""
    h = 1.0 / (n + 1)
    xs = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u = (16.0 * X * (1 - X) * Y * (1 - Y)).ravel()

    A = sp.identity(n * n).tocsc() - dt * laplacian_2d(n, h)
    solve = spla.factorized(A)

    nsteps = int(round(T / dt))
    thr = lam * dt

    t_extinct = None          # first time u is exactly the zero vector
    ever_nonzero_after = False
    times, norms = [], []
    cell = h  # for L2 norm: ||u||_L2 ~ h * ||u||_2 (2D)

    for k in range(1, nsteps + 1):
        v = solve(u)
        u = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
        t = k * dt
        times.append(t)
        norms.append(cell * np.linalg.norm(u))
        if t_extinct is None and not np.any(u):
            t_extinct = t
        elif t_extinct is not None and np.any(u):
            ever_nonzero_after = True

    persistent = (t_extinct is not None) and (not ever_nonzero_after)
    return t_extinct, persistent, np.array(times), np.array(norms)


def main():
    lams = [0.4, 0.6, 0.8]
    # (n, dt) refinement ladder: baseline of the paper first
    configs = [
        (49, 5e-4, "baseline (paper)"),
        (49, 2.5e-4, "dt/2"),
        (49, 1.25e-4, "dt/4"),
        (97, 5e-4, "mesh x2"),
        (97, 2.5e-4, "mesh x2, dt/2"),
        (193, 2.5e-4, "mesh x4, dt/2"),
    ]

    results = {}
    print(f"{'lambda':>7} {'n':>5} {'dt':>10} {'t*':>9} {'persist':>8} "
          f"{'wall[s]':>8}  tag")
    for lam in lams:
        results[lam] = []
        for n, dt, tag in configs:
            t0 = time.time()
            t_ext, pers, _, _ = run_reference(lam, n, dt)
            wall = time.time() - t0
            results[lam].append(dict(n=n, dt=dt, t_star=t_ext,
                                     persistent=bool(pers), tag=tag))
            print(f"{lam:7.1f} {n:5d} {dt:10.2e} "
                  f"{(t_ext if t_ext is not None else float('nan')):9.5f} "
                  f"{str(pers):>8} {wall:8.2f}  {tag}")

    with open("convergence_results_63.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary: spread of t* per lambda across the ladder
    print("\nSpread of detected t* per lambda (max - min over the ladder):")
    for lam in lams:
        ts = [r["t_star"] for r in results[lam] if r["t_star"] is not None]
        base = results[lam][0]["t_star"]
        fine = results[lam][-1]["t_star"]
        print(f"  lambda={lam}: baseline={base:.5f}  finest={fine:.5f}  "
              f"spread={max(ts)-min(ts):.5f}  "
              f"|baseline-finest|={abs(base-fine):.5f}")


if __name__ == "__main__":
    main()
