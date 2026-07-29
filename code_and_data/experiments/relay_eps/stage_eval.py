"""Stage 3 (GPU, cheap; decoupled from training per the staged-pipeline
policy): evaluate every trained run against the matching reference.

Outputs per run:  results/relay_eps/runs/<id>/eval.json + eval_arrays.npz
Aggregate:        results/relay_eps/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dr_pinns.relay_eps.metrics import evaluate_run


def load_reference(refdir: Path, lam: float) -> dict:
    d = np.load(refdir / f"ref_lam{lam:g}.npz")
    return dict(times=d["times"], l2=d["l2"], t_star=float(d["t_star"]),
                v1=d["v1"], v1_inf=float(d["v1_inf"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relay_eps.yaml")
    ap.add_argument("--runroot", default="results/relay_eps/runs")
    ap.add_argument("--refdir", default="results/relay_eps/reference")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    refdir = Path(args.refdir)
    results = []
    for run_dir in sorted(Path(args.runroot).iterdir()):
        if not (run_dir / "weights.npz").exists():
            continue
        out_json = run_dir / "eval.json"
        if out_json.exists() and not args.force:
            results.append(json.loads(out_json.read_text()))
            continue
        run_cfg = json.loads((run_dir / "run_config.json").read_text())
        ref = load_reference(refdir, float(run_cfg["lam"]))
        print(f"evaluating {run_dir.name} ...", flush=True)
        results.append(evaluate_run(cfg, run_dir, ref, out_json))

    summary = Path(args.runroot).parent / "eval_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"{len(results)} runs -> {summary}")


if __name__ == "__main__":
    main()
