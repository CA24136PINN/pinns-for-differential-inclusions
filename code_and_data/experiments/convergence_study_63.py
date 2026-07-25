r"""Standalone cross-check of the reference-solver refinement study (Sec. 6.3).

The authoritative refinement study is part of the pipeline:
    python3 experiments/run_experiment_63.py --stage reference
which writes results/raw/exp63/reference_manifest.json and feeds
paper/generated/table_63_solver_convergence.tex via --stage tables.

This script is an INDEPENDENT re-implementation of the same splitting
scheme (implicit diffusion + exact soft-threshold prox), kept as a
cross-check: it must reproduce the pipeline's detected extinction times
exactly.  CPU-only, ~30 s.

Usage:
    python3 experiments/convergence_study_63.py
    python3 experiments/convergence_study_63.py --check   # compare with pipeline
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LAMBDAS = [0.4, 0.6, 0.8]
CONFIGS = [  # (n, dt, tag) -- keep in sync with REFINEMENT_CONFIGS
    (49, 5e-4, "baseline"),
    (49, 2.5e-4, "dt/2"),
    (49, 1.25e-4, "dt/4"),
    (97, 5e-4, "mesh x2"),
    (97, 2.5e-4, "mesh x2, dt/2"),
    (193, 2.5e-4, "mesh x4, dt/2"),
]
OUT_DEFAULT = "results/raw/exp63/convergence_ref.json"


def laplacian_2d(n, h):
    """5-point Laplacian on the n x n interior grid, Dirichlet BC."""
    e = np.ones(n)
    T = sp.diags([e[:-1], -2.0 * e, e[:-1]], [-1, 0, 1]) / h**2
    I = sp.identity(n)
    return (sp.kron(I, T) + sp.kron(T, I)).tocsc()


def run_reference(lam, n, dt, T=0.3):
    """Return (t_extinct, persistent_up_to_T)."""
    h = 1.0 / (n + 1)
    xs = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u = (16.0 * X * (1 - X) * Y * (1 - Y)).ravel()

    A = sp.identity(n * n).tocsc() - dt * laplacian_2d(n, h)
    solve = spla.factorized(A)

    nsteps = int(round(T / dt))
    thr = lam * dt
    t_extinct = None
    ever_nonzero_after = False

    for k in range(1, nsteps + 1):
        v = solve(u)
        u = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
        if t_extinct is None:
            if not np.any(u):
                t_extinct = k * dt
        elif np.any(u):
            ever_nonzero_after = True

    return t_extinct, (t_extinct is not None) and (not ever_nonzero_after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--horizon", type=float, default=0.3)
    ap.add_argument("--check", action="store_true",
                    help="compare detected t* with the pipeline's "
                         "reference_manifest.json (exit 1 on mismatch)")
    args = ap.parse_args()

    results = {}
    print(f"{'lambda':>7} {'n':>5} {'dt':>10} {'t*':>9} {'persist':>8} "
          f"{'wall[s]':>8}  tag")
    for lam in LAMBDAS:
        results[lam] = []
        for n, dt, tag in CONFIGS:
            t0 = time.time()
            t_ext, pers = run_reference(lam, n, dt, T=args.horizon)
            wall = time.time() - t0
            results[lam].append(dict(n=n, dt=dt, t_star=t_ext,
                                     persistent=bool(pers), tag=tag))
            print(f"{lam:7.1f} {n:5d} {dt:10.2e} "
                  f"{(t_ext if t_ext is not None else float('nan')):9.5f} "
                  f"{str(pers):>8} {wall:8.2f}  {tag}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest = dict(
        description="Standalone cross-check of the Sec. 6.3 reference-"
                    "solver refinement study",
        horizon=args.horizon,
        lambdas=LAMBDAS,
        configs=[dict(n=n, dt=dt, tag=tag) for n, dt, tag in CONFIGS],
        results={str(k): v for k, v in results.items()},
    )
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {args.out}")

    if args.check:
        ref_path = "results/raw/exp63/reference_manifest.json"
        with open(ref_path) as f:
            pipe = json.load(f)["refinement"]
        ok = True
        for lam in LAMBDAS:
            pipe_ts = sorted(
                round(e["t_ext"], 10)
                for e in pipe[str(lam)].values() if e["t_ext"] is not None)
            mine_ts = sorted(
                round(r["t_star"], 10)
                for r in results[lam] if r["t_star"] is not None)
            match = pipe_ts == mine_ts
            ok &= match
            print(f"  check lambda={lam}: "
                  f"{'MATCH' if match else 'MISMATCH'}")
        if not ok:
            raise SystemExit(1)
        print("Cross-check against the pipeline: all t* match.")


if __name__ == "__main__":
    main()
