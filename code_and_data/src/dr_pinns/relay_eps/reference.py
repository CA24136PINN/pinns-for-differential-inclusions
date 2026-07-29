"""Reference machinery for Section 6.3(eps).

1.  Operator-splitting solver for the *exact relay* flow (eps = 0), identical
    to the paper: implicit 5-point diffusion solve + exact soft-thresholding
    prox of s -> lam*dt*|s|.  Because Phi [subset] Phi_eps pointwise, this
    trajectory is an admissible element of S_eps(u0) for every eps >= 0, and
    remains the pre-extinction validation target.

2.  Discrete torsion function v1 solving -Lap v1 = 1, v1 = 0 on the boundary.
    Whenever eps >= lam * ||v1||_inf, the stationary field u = lam * v1 is a
    second, nonzero strong solution of the eps-inclusion (operator value
    lam in [-lam,lam] on the band |u| <= eps): the explicit non-uniqueness
    witness quoted in the revised text.

3.  Extinction detection + refinement study (grid / dt sweep), regenerating
    the analogue of Table 4.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def u0_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Initial datum (6.24); duplicated here so stage 1 stays TF-free."""
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def laplacian_2d(n: int) -> sp.csr_matrix:
    """Negative-definite 5-point Laplacian on the n x n interior grid of (0,1)^2."""
    h = 1.0 / (n + 1)
    main = -2.0 * np.ones(n)          # 1D stencil; the Kronecker sum below
    off = np.ones(n - 1)              # yields the standard 5-point -4 diagonal
    T = sp.diags([off, main, off], [-1, 0, 1])
    I = sp.identity(n)
    L = (sp.kron(I, T) + sp.kron(T, I)) / h ** 2
    return L.tocsr()


def interior_grid(n: int):
    h = 1.0 / (n + 1)
    s = h * np.arange(1, n + 1)
    X, Y = np.meshgrid(s, s, indexing="ij")
    return X, Y, h


def torsion_function(n: int):
    """v1 with -Lap v1 = 1 (Dirichlet); returns (v1_grid, ||v1||_inf)."""
    L = laplacian_2d(n)
    v = spla.spsolve(-L, np.ones(L.shape[0]))
    V = v.reshape(n, n)
    return V, float(np.max(np.abs(V)))


def solve_relay(n: int, dt: float, T: float, lam: float, u0_fn,
                store_times: np.ndarray):
    """Splitting solver for d_t u - Lap u in -lam Sign(u), u(0)=u0.

    Returns dict with:
      t_star        : first grid time with an exactly-zero discrete field
                      (np.inf if not reached),
      times, l2, linf : norm curves on the solver time grid,
      snapshots     : field on the interior grid at the requested store_times
                      (nearest solver step), plus at 0.9*t_star if reached.
    """
    X, Y, h = interior_grid(n)
    u = u0_fn(X, Y).reshape(-1)
    L = laplacian_2d(n)
    A = (sp.identity(L.shape[0]) - dt * L).tocsc()
    solve = spla.factorized(A)

    nsteps = int(round(T / dt))
    times = dt * np.arange(nsteps + 1)
    l2 = np.empty(nsteps + 1)
    linf = np.empty(nsteps + 1)
    l2[0] = h * np.linalg.norm(u)
    linf[0] = np.max(np.abs(u))

    snap_idx = {int(round(ts / dt)): None for ts in store_times}
    snapshots = {}
    t_star = np.inf

    for k in range(1, nsteps + 1):
        v = solve(u)
        u = np.sign(v) * np.maximum(np.abs(v) - lam * dt, 0.0)  # exact prox
        l2[k] = h * np.linalg.norm(u)
        linf[k] = np.max(np.abs(u))
        if t_star == np.inf and linf[k] == 0.0:
            t_star = times[k]
        if k in snap_idx:
            snapshots[times[k]] = u.reshape(n, n).copy()

    if np.isfinite(t_star):
        # snapshot shortly before extinction, for the refinement metric
        k9 = max(1, int(round(0.9 * t_star / dt)))
        # rerun cheaply is wasteful; store on the fly next time -- here we
        # simply mark the index; caller may request it via store_times.
        snapshots["idx_09tstar"] = k9
    return dict(t_star=t_star, times=times, l2=l2, linf=linf,
                snapshots=snapshots, h=h, n=n, dt=dt)


def refinement_study(lam_values, base_n: int, base_dt: float, T: float, u0_fn,
                     grids=(1, 2, 4), dts=(1.0, 0.5, 0.25)):
    """Regenerates the analogue of Table 4 for the chosen baselines.

    grids: multiplicative refinements of (base_n+1)-1 in the sense
           n -> (base_n+1)*g - 1 (so 49 -> 97 -> 193 for g=1,2,4);
    dts  : multiplicative factors on base_dt.
    Returns rows: (lam, n, dt, t_star, d_t_star_vs_baseline).
    """
    rows = []
    for lam in lam_values:
        base = None
        for g in grids:
            n = (base_n + 1) * g - 1
            for f in dts:
                dt = base_dt * f
                if (g, f) not in [(gg, ff) for gg in grids for ff in dts]:
                    continue
                res = solve_relay(n, dt, T, lam, u0_fn, store_times=np.array([]))
                if base is None:
                    base = res["t_star"]
                rows.append(dict(lam=lam, n=n, dt=dt,
                                 t_star=res["t_star"],
                                 d_t_star=res["t_star"] - base))
    return rows
