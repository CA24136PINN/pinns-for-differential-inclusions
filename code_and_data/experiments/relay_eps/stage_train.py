"""Stage 2 (GPU, expensive): train one DR-PINN run identified by --run-id.

Launched in parallel across GPUs by scripts/launch_relay_eps.sh; one process
per run, one run per GPU at a time. Skips runs whose weights already exist
(safe restart after interruption).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dr_pinns.relay_eps.training import train_run
from run_matrix import build_matrix  # noqa: E402 (same directory)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/relay_eps.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--outroot", default="results/relay_eps/runs")
    ap.add_argument("--smoke", action="store_true",
                    help="200 epochs, small pools; pipeline check only")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg = dict(cfg, epochs=200, refresh_every=50, candidate_size=2000,
                   pool_size=400, val_every=50)

    runs = {r["id"]: r for r in build_matrix(cfg)}
    if args.run_id not in runs:
        sys.exit(f"unknown run id: {args.run_id}")
    run_cfg = runs[args.run_id]

    outdir = Path(args.outroot) / args.run_id
    if (outdir / "weights.npz").exists():
        print(f"[{args.run_id}] weights exist, skipping (delete to retrain)")
        return
    train_run(cfg, run_cfg, outdir)


if __name__ == "__main__":
    main()
