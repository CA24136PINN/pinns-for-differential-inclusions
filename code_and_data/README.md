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

The full Example 6.3 run trains four networks for 40 000 epochs each
(≈6 h wall clock; GPU strongly recommended). To make that cost a one-time
cost, every experiment script is staged:

```bash
python3 experiments/run_experiment_63.py --stage train     # hours: computes
                                                           # everything, writes
                                                           # results/raw/exp63/
python3 experiments/run_experiment_63.py --stage tables    # seconds: macros
python3 experiments/run_experiment_63.py --stage figures   # seconds: figures
```

`--stage train` stores **every array any figure needs** in
`results/raw/exp6X/raw_data.npz` and **every number any macro needs** in
`results/raw/exp6X/manifest_6X.json`, plus the network checkpoints. The
`figures` and `tables` stages only read those files — figures can be
restyled, re-zoomed or recolored arbitrarily many times without retraining.
`scripts/make_figures.sh` and `scripts/make_tables.sh` simply loop this over
all three experiments. `--smoke` (accepted by every script and by
`reproduce_all.sh`) runs the identical pipeline with truncated iteration
budgets for an end-to-end test; smoke results are not publication-grade.

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

| Experiment | Stage `train` | Hardware |
|---|---|---|
| 6.1 linear control + Quickhull | ~30–60 min | CPU |
| 6.2 rotating ellipse + companion | ~10–15 min | CPU |
| 6.3 relay parabolic (4 × 40k epochs) | **~6 h** | GPU recommended |
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
