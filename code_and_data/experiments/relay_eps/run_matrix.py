"""Enumerate the full run matrix for the revised Section 6.3.

Groups:
  A. eps-sweep   : lam = lam_baseline, eps in eps_sweep, sampler = front
  B. lam-sweep   : eps = eps_baseline, lam in lam_sweep, sampler = front
  C. ablation    : (lam_baseline, eps_baseline), sampler in ablation_samplers

Run id format: lam{lam}_eps{eps}_s{seed}_{sampler}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def run_id(lam, eps, seed, sampler) -> str:
    return f"lam{lam:g}_eps{eps:g}_s{seed}_{sampler}"


def build_matrix(cfg: dict) -> list[dict]:
    runs = []
    lam0, eps0 = cfg["lam_baseline"], cfg["eps_baseline"]
    for eps in cfg["eps_sweep"]:
        for seed in cfg["seeds"]:
            runs.append(dict(lam=lam0, eps=eps, seed=seed, sampler="front"))
    for lam in cfg["lam_sweep"]:
        for seed in cfg["seeds"]:
            runs.append(dict(lam=lam, eps=eps0, seed=seed, sampler="front"))
    for sampler in cfg["ablation_samplers"]:
        for seed in cfg["seeds"]:
            runs.append(dict(lam=lam0, eps=eps0, seed=seed, sampler=sampler))
    for r in runs:
        r["id"] = run_id(r["lam"], r["eps"], r["seed"], r["sampler"])
    # de-duplicate (eps-sweep at eps0 overlaps nothing; keep defensive check)
    seen, uniq = set(), []
    for r in runs:
        if r["id"] not in seen:
            seen.add(r["id"]); uniq.append(r)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relay_eps.yaml")
    ap.add_argument("--list", action="store_true", help="print run ids")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    runs = build_matrix(cfg)
    if args.list:
        for r in runs:
            print(r["id"])
    else:
        print(f"{len(runs)} runs", file=sys.stderr)
        for r in runs:
            print(r)


if __name__ == "__main__":
    main()
