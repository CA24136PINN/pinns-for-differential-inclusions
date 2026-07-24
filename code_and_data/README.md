# DR-PINNs — replication package

Code and data for the DR-PINN manuscript (Sections 6.1–6.3): distance-residual
physics-informed neural networks for differential inclusions.

## Quick start

```bash
conda env create -f environment.yml
conda activate dr-pinn
bash scripts/system_info.sh
bash scripts/run_experiment_61.sh
bash scripts/run_experiment_62.sh
bash scripts/run_experiment_63.sh
bash scripts/make_tables.sh
bash scripts/make_figures.sh
bash scripts/build_paper.sh
```

or everything in one command:

```bash
bash reproduce_all.sh            # full run
bash reproduce_all.sh --smoke    # fast end-to-end pipeline test (~5–10 min)
```

`pip install -r requirements.txt` works as an alternative to conda.

## Directory layout

```
code_and_data/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── src/                    shared library code (paper_style, repro_utils)
├── experiments/            run_experiment_61/62/63.py (staged pipelines)
├── configs/                exp61/62/63.json — all hyperparameters
│                           (full + smoke overrides)
├── scripts/                bash entry points (see Quick start)
├── results/
│   ├── raw/                training outputs: raw_data.npz, manifest_6X.json,
│   │                       checkpoints/  (one subdir per experiment)
│   ├── figures/            all paper figures (PNG)
│   └── aggregated/         results_6X.tex (LaTeX macros), manifests,
│                           system_info.txt
├── paper/                  dr-pinns.tex + references.bib; figures/ and
│                           generated/ are synced from results/ by
│                           scripts/build_paper.sh
├── docs/                   review-response notes (mapping fixes → referee
│                           points, in Polish)
└── reproduce_all.sh
```

## Training is decoupled from figures and tables

Every experiment script is staged; training (hours) writes raw artifacts
once, and figures/tables regenerate from them in seconds, arbitrarily many
times, without retraining. For 6.1/6.2:
`--stage train` → `results/raw/exp6X/{raw_data.npz, manifest_6X.json}`,
then `--stage tables` / `--stage figures`.

Example 6.3 (referee revision) is a **pre-registered multi-run grid**:

```bash
python3 experiments/run_experiment_63.py --stage reference   # CPU, minutes:
                # reference solver + grid/time-step convergence study
python3 scripts/gpu_launcher.py                              # GPU, ~11 x 6 h:
                # 3 seeds x lambda in {0.4, 0.6, 0.8} (adaptive sampling)
                # + ablation for lambda = 0.6: uniform and residual-RAR
python3 experiments/run_experiment_63.py --stage analyze     # minutes, from
                # checkpoints: t_first / t_persistent per threshold, rebound,
                # plateau stats, residual distributions on 100k fresh points,
                # multi-seed aggregation (median + range)
python3 experiments/run_experiment_63.py --stage tables      # seconds
python3 experiments/run_experiment_63.py --stage figures     # seconds
```

`bash scripts/run_experiment_63.sh` chains reference → grid → analyze.
Seeds (11, 23, 47) are fixed in `configs/exp63.json` **before** looking at
any results; the network-initialisation seed and the collocation-sampling
seed are decoupled (`sample_seed = net_seed + 1000`). Every run lives in
its own directory `results/raw/exp63/runs/<tag>/` with checkpoint, curves,
log and `run.json`, and `--stage analyze` recomputes all reported metrics
from the checkpoints — postprocessing changes never require retraining.

### Dynamic multi-GPU scheduling (shared machine)

`scripts/gpu_launcher.py` distributes the grid over the available GPUs
**dynamically**: it polls `nvidia-smi` and only assigns a job to a card
with at least `--min-free-mb` (default 8000 MB) free, so co-existing
computations by other users are respected; each job runs with
`CUDA_VISIBLE_DEVICES=<idx>` and `TF_FORCE_GPU_ALLOW_GROWTH=true`, so
TensorFlow never grabs a whole 10 GB card. The scheduler is **resumable**:
finished runs are skipped, so it can be killed and restarted at any time
(also after adding seeds). On 4 free GPUs the full 11-run grid takes
roughly 3 x 6 h wall clock. `--smoke` (accepted everywhere, incl.
`reproduce_all.sh`) runs the identical pipeline with tiny budgets;
without any GPU the jobs fall back to sequential CPU execution.

### Mapping to the referee's points

* item 1.1 (terminology + definitions): `--stage analyze` reports
  `t_first(eps)`, `t_persistent(eps)` and `r_post` per run; macros/tables
  say "threshold-crossing time" and print
  "no persistent extinction before T" where applicable
  (`table_63_thresholds.tex`, `table_63_runs.tex`);
* item 1.2 (threshold sensitivity): eps in {5e-4, 1e-3, 2e-3, 5e-3},
  `table_63_thresholds.tex` + `threshold_sensitivity.png`;
* item 1.4 (residual distribution): mean/RMS/median/p90/p95/p99/max on an
  independent 100k-point set, split `|u_theta| < 0.01` (front) vs away:
  `table_63_residuals.tex` + `residual_distribution.png`;
* item 1.5 (solver convergence): 49²/97² x {5e-4, 2.5e-4}:
  `table_63_solver_convergence.tex` + `solver_convergence.png`;
* item 1.6 (plateau consistency): per-run median/min/max after t*_ref:
  `table_63_plateau.tex`;
* item 2 (multi-seed 6.3): 3 pre-registered seeds per lambda, median +
  full range everywhere, per-run table `table_63_runs.tex`;
* item 3 (sampling ablation): uniform + residual-RAR for lambda = 0.6,
  same seed (`RelayAbl*` macros);
* item 4 (6.1): per-seed table `table_61_seeds.tex`, seeds listed
  explicitly, finite-difference gradient check of the dist² projection
  loss (`LCGradCheck*` macros; restricted to parameters with
  non-negligible gradient — float32 FD cannot resolve |g| ≈ 0);
* item 5 (6.2): dense-residual median/p95/p99, Newton iteration count and
  measured final secular residual (`EllipNewton*` macros).

The legacy macro names used by the current manuscript are still emitted
(filled with the honest multi-seed / representative-run values), so the
paper compiles before the section text is rewritten.

## Text/figure/number synchronization

Every number quoted in the paper is a LaTeX macro generated from a single
documented run: `paper/dr-pinns.tex` loads
`generated/results_6{1,2,3}.tex` via `\InputIfFileExists`; missing macros
fall back to red `[TBD: run 6.x]` markers, and `scripts/build_paper.sh`
exits with a nonzero code if any `[TBD:]` survives in the compiled PDF
(CI-friendly). Each `manifest_6X.json` records seeds, library versions,
hardware, runtimes and all reported values; the Example 6.2/6.3 checkpoints
make every network-derived figure reproducible bit-for-bit from the saved
weights.

## Runtimes (orientation)

| Experiment | Stage | Hardware |
|---|---|---|
| 6.1 linear control + Quickhull | ~30–60 min | CPU |
| 6.2 rotating ellipse + companion | ~10–15 min | CPU |
| 6.3 reference + convergence study | minutes | CPU |
| 6.3 training grid (11 runs × 40k epochs) | **~66 GPU-h** (≈3×6 h on 4 free GPUs) | GPU |
| 6.3 analyze (from checkpoints) | minutes | CPU/GPU |
| tables + figures + paper | seconds–minutes | CPU |

`bash scripts/system_info.sh` records OS, CPU, core count, RAM, GPU(s),
CUDA/driver, Python and library versions to
`results/aggregated/system_info.txt`, so it is always documented what the
shipped numbers were produced on.

## Provenance of shipped artifacts

The `results/` content in this package comes from a **smoke run** and serves
only to verify the pipeline end-to-end; for submission, overwrite it with a
full run (`bash reproduce_all.sh`). The mapping of code changes to the
referee's points is documented in `docs/README_review_response_pl.md`.
