"""Stage 1 (cheap, CPU): reference relay solutions, refinement study, torsion
witness. Regenerated in the same execution as everything else, per the
replication-package policy of the paper.

Outputs:
  results/relay_eps/reference/ref_lam{lam}.npz    (times, l2, linf, t_star)
  results/relay_eps/reference/torsion.npz         (v1 grid, v1_inf)
  results/relay_eps/reference/refinement.csv      (Table-4 analogue)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dr_pinns.relay_eps.reference import (refinement_study, solve_relay,
                                          torsion_function, u0_np)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relay_eps.yaml")
    ap.add_argument("--outdir", default="results/relay_eps/reference")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    n, dt, T = cfg["ref_n"], cfg["ref_dt"], cfg["T"]
    if args.smoke:
        n, dt = 25, 2e-3

    # torsion function / non-uniqueness witness
    v1, v1_inf = torsion_function(n)
    np.savez(out / "torsion.npz", v1=v1, v1_inf=v1_inf)
    lam0 = cfg["lam_baseline"]
    print(f"||v1||_inf = {v1_inf:.4f};  lam*||v1||_inf = {lam0 * v1_inf:.4f} "
          f"(witness admissible iff eps >= this value)")

    for lam in cfg["ref_lams"]:
        res = solve_relay(n, dt, T, lam, u0_np, store_times=np.array([T]))
        np.savez(out / f"ref_lam{lam:g}.npz",
                 times=res["times"], l2=res["l2"], linf=res["linf"],
                 t_star=res["t_star"], v1=v1, v1_inf=v1_inf)
        ts = res["t_star"]
        print(f"lam={lam:g}: t* = {ts if np.isfinite(ts) else 'not reached'}")

    if not args.smoke:
        rows = refinement_study([l for l in cfg["ref_lams"] if l > 0],
                                n, dt, T, u0_np,
                                grids=cfg["ref_refine_grids"],
                                dts=cfg["ref_refine_dts"])
        with open(out / "refinement.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["lam", "n", "dt", "t_star", "d_t_star"])
            w.writeheader(); w.writerows(rows)
        print(f"refinement study: {len(rows)} rows -> {out/'refinement.csv'}")


if __name__ == "__main__":
    main()
