# PINNs for Differential Inclusions

**Distance-Residual Physics-Informed Neural Networks (DR-PINNs) for differential and partial differential inclusions.**

This repository contains the manuscript, a fully staged replication package
for all numerical examples of the paper (Sections 6.1–6.3), and an archive of
earlier exploratory material.

> **Project status:** active research repository, organized around the
> replication package in `code_and_data/`. It is not an installable Python
> library.

## Overview

Classical PINNs assume that the governing law is an equality,

$$
D(u)=f.
$$

Differential inclusions replace the single right-hand side by a set of
admissible values,

$$
D(u)\in F(u).
$$

For example, an ordinary differential inclusion has the form

$$
\dot{x}(t)\in F(t,x(t)), \qquad x(0)=x_0.
$$

Because there is no unique value to subtract from $D(u)$, the classical
pointwise residual is replaced by the squared distance to the admissible set:

$$
R(u_\theta)(z) = \text{dist}^2\left(D(u_\theta)(z),F(u_\theta)(z)\right).
$$

The corresponding DR-PINN loss is

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}
\text{dist}^2\left(D(u_\theta)(z_i),F(u_\theta)(z_i)\right).
$$

The residual vanishes exactly when the differential inclusion is satisfied.
For closed convex admissible sets, the metric projection is unique and

$$
\nabla_v\text{dist}^2(v,C) = 2\bigl(v-\Pi_C(v)\bigr),
$$

which makes the loss suitable for gradient-based optimization. If the
admissible set depends on the network state, that dependence must remain in
the computational graph (see Remark 4.2 of the manuscript); the
implementations evaluate translated residuals such as
$\text{dist}^2(\dot x_\theta - A x_\theta, B\widetilde U)$ and detach
the entire numerically computed projection point in the backward pass.

## Repository layout

```text
pinns-for-differential-inclusions/
├── README.md                 # this file
├── LICENSE                   # Apache 2.0
├── sync_update.sh            # utility: apply an update bundle into
│                             # code_and_data/ (whitelist, dry-run by
│                             # default, backups; never touches GPU outputs)
├── code_and_data/            # REPLICATION PACKAGE (single source of truth)
│   ├── README.md             # detailed package documentation
│   ├── environment.yml, requirements.txt
│   ├── reproduce_all.sh      # full replication in one command (see below)
│   ├── configs/              # exp61/62/63.json — all hyperparameters
│   ├── experiments/          # staged pipelines:
│   │   ├── run_experiment_61.py      Sec. 6.1 (linear control, polytope)
│   │   ├── run_experiment_62.py      Sec. 6.2 (rotating ellipse)
│   │   ├── run_experiment_63.py      Sec. 6.3 (relay parabolic inclusion)
│   │   └── convergence_study_63.py   standalone cross-check of the
│   │                                 reference-solver refinement study
│   ├── scripts/              # bash entry points, GPU scheduler,
│   │                         # make_tables / make_figures / build_paper
│   ├── src/                  # paper_style.py, repro_utils.py
│   ├── results/              # raw/ (training outputs, checkpoints),
│   │                         # figures/ (all paper PNGs),
│   │                         # aggregated/ (LaTeX macros + tables)
│   └── paper/                # dr-pinns.tex/.pdf, references.bib;
│                             # figures/ and generated/ are synced from
│                             # results/ by scripts/build_paper.sh
└── old_stuff/                # ARCHIVE of the earlier repository layout:
    ├── code/                 # original notebooks + first experiment code
    ├── notes/                # TeX notes: convex rotating-ellipse and
    │                         # nonconvex two-disk inclusions
    ├── replication_package/  # standalone geometry benchmarks (Plotly 3D
    │                         # velocity tubes; run via their run_export.sh)
    ├── paper/                # earlier manuscript snapshot
    └── scripts/              # earlier reproduce.sh entry point
```

Everything the manuscript reports — every number, table, and figure — is
produced by `code_and_data/`. The archive in `old_stuff/` is kept for
provenance (early notebooks, hand-crafted geometry illustrations, and the
nonconvex two-disk notes on relaxation/convexification of weak limits,
$\dot x\in\text{co}F(t,x)$); it is not part of the replication
pipeline.

## Reproducing the paper

From `code_and_data/`:

```bash
conda env create -f environment.yml && conda activate dr-pinn
bash reproduce_all.sh            # full run (Sec. 6.3 grid: ~66 GPU-h;
                                 # ~3 x 6 h wall clock on 4 GPUs)
bash reproduce_all.sh --smoke    # end-to-end pipeline test (~5–10 min)
```

or stage by stage (`scripts/run_experiment_6{1,2,3}.sh`, then
`make_tables.sh`, `make_figures.sh`, `build_paper.sh`). Training (hours) is
strictly decoupled from tables and figures (seconds): all reported metrics
are recomputed from saved checkpoints and manifests, so postprocessing
changes never require retraining. See `code_and_data/README.md` for the
staged design, the pre-registered multi-seed grid of Section 6.3, the
dynamic multi-GPU scheduler, and the mapping of each pipeline output to the
referee's points.

## Paper experiments (Section 6 of the manuscript)

| # | Manuscript section | Pipeline | Figures (`paper/figures/`) |
|---|--------------------|----------|-----------------------------|
| 1 | 6.1 Linear control system with polytopic input set | `experiments/run_experiment_61.py` | `control_set`, `loss_single`, `trajectory_vs_tube`, `ensemble_hausdorff`, `scalability` |
| 2 | 6.2 Planar inclusion with rotating ellipsoidal constraint | `experiments/run_experiment_62.py` (hard-admissible selector + endpoint-hard DR-PINN companion) | `ellipse_velocity_tube`, `ellipse_selector_level`, `ellipse_state_trajectory`, `drpinn_companion_seed0`, `companion_diagnostics` |
| 3 | 6.3 Reaction–diffusion inclusion with relay feedback | `experiments/run_experiment_63.py` (stages: `reference` → GPU grid via `scripts/gpu_launcher.py` → `analyze` → `tables`/`figures`) | `reference_extinction_curves`, `training_history_relay`, `dr_pinn_vs_reference`, `branch_selection_diagnostic`, `extinction_time_sweep`, `threshold_sensitivity`, `residual_distribution`, `solver_convergence` |

All figures are exported through the shared style in `src/paper_style.py`
(Computer Modern serif, dpi=180) directly by the pipeline; `paper/figures/`
and `paper/generated/` (LaTeX macros and ready-made tabulars) are updated
exclusively by `scripts/build_paper.sh`, which also compiles the manuscript
and fails if any `[TBD:]` placeholder survives in the PDF.

### Reference-solver convergence study (Sec. 6.3)

The extinction times of the relay benchmark are validated by a refinement
study of the reference splitting solver (mesh $49^2\to97^2\to193^2$, time
step $\Delta t\to\Delta t/2\to\Delta t/4$), computed in the `reference`
stage of experiment 6.3 and reported in
`table_63_solver_convergence.tex`. An independent re-implementation is kept
as a cross-check and must reproduce the pipeline's detected $t^*$ exactly:

```bash
python3 experiments/convergence_study_63.py --check
```

## Requirements

- Python **3.10+**, TensorFlow (all three experiments), NumPy, SciPy,
  Matplotlib; see `code_and_data/environment.yml` /
  `code_and_data/requirements.txt` for pinned versions.
- GPU strongly recommended for the Section 6.3 training grid (the scheduler
  falls back to sequential CPU execution, and `--smoke` runs everywhere).
- A Unix-like shell for the provided scripts.

The archived geometry benchmarks in `old_stuff/replication_package/`
additionally use Plotly + Kaleido and create their own virtual environments
via the local `run_export.sh` scripts.

## Reproducibility notes

- Seeds for the Section 6.3 grid (11, 23, 47) are fixed in
  `configs/exp63.json` before looking at any results; network and sampling
  seeds are decoupled.
- Every training run lives in its own directory
  `results/raw/exp63/runs/<tag>/` with checkpoint, curves, log, and
  `run.json`; the GPU scheduler is resumable (finished runs are skipped).
- `results/aggregated/system_info.txt` records the hardware and library
  versions of the reported run.
- `sync_update.sh` applies incremental update bundles into
  `code_and_data/` safely: dry run by default, explicit whitelist, backup
  of every overwritten file, and a hard guard that refuses to touch
  `results/raw/exp63/runs/`, `checkpoints/`, or the analysis arrays.

## Mathematical scope

The associated research develops the distance-residual principle for:

- ordinary differential inclusions and controlled systems represented as
  differential inclusions;
- parabolic partial differential inclusions with set-valued reaction terms;
- convex admissible sets, where projection and consistency theory are
  strongest (consistency theorems for both the ODE and the parabolic case);
- nonconvex admissible sets, where relaxation, chattering, projection
  nonuniqueness, and strong-versus-weak convergence must be distinguished
  (exploratory material in `old_stuff/notes/`).

## Contributing

Please develop changes on a separate branch rather than committing directly
to `main`:

```bash
git checkout main && git pull
git checkout -b feature/short-description
```

After making and testing the changes, open a pull request and briefly
describe: (1) the mathematical or numerical change; (2) how it was tested;
(3) which outputs were regenerated; (4) whether it affects the assumptions
or interpretation of an experiment. Regenerated numbers must come from the
pipeline (`make_tables.sh` / `make_figures.sh`), never be edited by hand.

## Citation

The repository accompanies the working preprint. A provisional BibTeX entry:

```bibtex
@misc{filipkovska2026drpinns,
  title  = {Distance-Residual Physics-Informed Neural Networks:
            A Deep Learning Framework for Differential and
            Partial Differential Inclusions},
  author = {Filipkovska, Maria and Mar{\'i}n, Juan Jos{\'e} and
            Oner, Isil and Periago, Francisco and Rykaczewski, Krzysztof},
  year   = {2026},
  note   = {Working preprint}
}
```

## License

This repository is distributed under the [Apache License 2.0](LICENSE).
