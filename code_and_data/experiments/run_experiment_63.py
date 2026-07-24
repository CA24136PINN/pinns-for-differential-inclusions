"""
Section 6.3 replication script: reaction-diffusion inclusion with relay
(sliding-mode) feedback.

Single-run, clean-state pipeline. One execution produces, in a single
documented run:

  figures/  reference_extinction_curves.png, training_history_relay.png,
            dr_pinn_vs_reference.png, branch_selection_diagnostic.png,
            extinction_time_sweep.png
  checkpoints/ relay_lam*.weights.h5 -- the exact networks behind every figure
  generated/results_63.tex   -- every number quoted in Section 6.3, as macros
  generated/manifest_63.json -- seeds, versions, hardware and reported values

Fixes relative to the previous notebook (referee report, Example 6.3):

  (F1) The branch selection in the loss uses the EXACT relay multifunction
       Phi: the interval branch [-lambda, lambda] is applied only at exact
       floating-point zero (eps = 0).  The regularized Phi_eps of the previous
       version is gone from training; the Phi_eps discussion is replaced by a
       cheap POST-TRAINING sensitivity check (F2).
  (F2) eps-sensitivity check: after training, the distance residual of the
       trained network is re-evaluated on one fixed validation set for
       eps in {0, 1e-8, 1e-6, 1e-4}; the maximum relative deviation from the
       eps = 0 value is exported as a macro (expected: 0 to machine
       precision, since the network essentially never outputs u_theta = 0
       exactly and the branch sets coincide off {0 < |s| <= eps}).
  (F3) Every figure and every quoted number comes from the same run and the
       same checkpoint: the branch-selection diagnostic (Figure 11) is
       generated from the identical in-memory network used for validation,
       and the weights are saved so the figure can be regenerated bit-for-bit.
  (F4) All quantitative statements exported as LaTeX macros (same mechanism
       as Section 6.1).

Usage:  python run_experiment_63.py            # full run (GPU recommended)
        python run_experiment_63.py --smoke    # tiny smoke test (~2 min CPU)
"""

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import repro_utils as R  # noqa: E402
from paper_style import apply_paper_style  # noqa: E402

apply_paper_style()

# ----------------------------------------------------------------------------
# Problem data
# ----------------------------------------------------------------------------

LAMBDA_BASELINE = 0.6
LAMBDA_SWEEP_REF = [0.0, 0.4, 0.6, 0.8]  # 0.0 = heat-equation control case
LAMBDA_SWEEP_MAIN = [0.4, 0.6, 0.8]
LAMBDA_COLORS = {0.0: "0.45", 0.4: "tab:blue", 0.6: "tab:orange",
                 0.8: "tab:green"}
T_HORIZON = 0.3
REF_DT = 5e-4
REF_N = 49
NORM_FLOOR = 1e-6
EXTINCTION_TOL = 1e-3  # fixed tolerance (the choice reported in the paper)
EVAL_N_TIMES = 121     # evaluation-time grid step = T/120 = 2.5e-3


def u0_numpy(X, Y):
    return 16.0 * X * (1 - X) * Y * (1 - Y)


def u0_tf(x, y):
    return 16.0 * x * (1 - x) * y * (1 - y)


# ----------------------------------------------------------------------------
# Reference solver (implicit diffusion + exact proximal soft-thresholding)
# ----------------------------------------------------------------------------


def build_laplacian(N):
    h = 1.0 / (N + 1)
    e = np.ones(N)
    L1 = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1]) / h**2
    I1 = sp.identity(N)
    return (sp.kron(L1, I1) + sp.kron(I1, L1)).tocsc(), h


def soft_threshold(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def run_reference_solver(lam, T=T_HORIZON, dt=REF_DT, N=REF_N):
    Lap, h = build_laplacian(N)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    n_steps = int(round(T / dt))
    Asys = (sp.identity(N * N, format="csc") - dt * Lap).tocsc()
    solve = spla.factorized(Asys)
    u = u0_numpy(X, Y).flatten()
    times = [0.0]
    l2norms = [np.sqrt(np.sum(u**2) * h**2)]
    extinction_time = None
    for n in range(1, n_steps + 1):
        v = solve(u)
        u = soft_threshold(v, dt * lam)
        nrm = np.sqrt(np.sum(u**2) * h**2)
        times.append(n * dt)
        l2norms.append(nrm)
        if extinction_time is None and nrm == 0.0:
            extinction_time = n * dt
    return (np.array(times), np.array(l2norms), u.reshape(N, N),
            extinction_time, X, Y)


# ----------------------------------------------------------------------------
# DR-PINN: ansatz, exact-Phi loss (F1), adaptive sampling, hold-then-decay LR
# ----------------------------------------------------------------------------


def build_mlp(in_dim=3, hidden=128, n_hidden_layers=5, out_dim=1):
    inputs = tf.keras.Input(shape=(in_dim,))
    h = inputs
    for _ in range(n_hidden_layers):
        h = tf.keras.layers.Dense(
            hidden, activation="tanh",
            kernel_initializer="glorot_normal", bias_initializer="zeros",
        )(h)
    outputs = tf.keras.layers.Dense(
        out_dim, kernel_initializer="glorot_normal", bias_initializer="zeros"
    )(h)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def u_theta(net, t, x, y):
    txy = tf.concat([t, x, y], axis=1)
    B = x * (1 - x) * y * (1 - y)
    return u0_tf(x, y) + t * B * net(txy)


def relay_distance_squared(z, s, lam, eps=0.0):
    """dist^2(z, Phi_eps(s)) with Phi_eps(s) = -lam Sign_eps(s).

    With the default eps = 0.0 this is the distance to the EXACT relay
    multifunction Phi of the manuscript (F1): the interval branch is applied
    only at exact floating-point zero, i.e. on a set the smooth ansatz hits
    with probability zero, so a.e. the pure sign branches are used.
    eps > 0 is retained ONLY for the post-training sensitivity check (F2);
    it is never used during training.
    """
    pos = s > eps
    neg = s < -eps
    d_pos = (z + lam) ** 2
    d_neg = (z - lam) ** 2
    d_zero = tf.maximum(tf.abs(z) - lam, 0.0) ** 2
    return tf.where(pos, d_pos, tf.where(neg, d_neg, d_zero))


def pde_operator(net, t, x, y):
    """z = d_t u - Delta u by automatic differentiation."""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([t, x, y])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([t, x, y])
            u = u_theta(net, t, x, y)
        u_t = tape1.gradient(u, t)
        u_x = tape1.gradient(u, x)
        u_y = tape1.gradient(u, y)
    u_xx = tape2.gradient(u_x, x)
    u_yy = tape2.gradient(u_y, y)
    del tape1, tape2
    return u, u_t - (u_xx + u_yy)


class HoldThenExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, hold_steps, decay_steps,
                 decay_rate):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        hold = tf.cast(self.hold_steps, tf.float32)
        decayed = self.initial_learning_rate * tf.pow(
            self.decay_rate, (step - hold) / self.decay_steps
        )
        return tf.where(step < hold, self.initial_learning_rate, decayed)

    def get_config(self):
        return dict(initial_learning_rate=self.initial_learning_rate,
                    hold_steps=self.hold_steps, decay_steps=self.decay_steps,
                    decay_rate=self.decay_rate)


def train_dr_pinn(lam, T, n_epochs=40000, n_colloc=2000, lr=2e-3,
                  hidden=128, n_hidden_layers=5, log_every=2000,
                  val_every=250, n_val=4000, seed=0, sample_seed=None,
                  hold_steps=5000, decay_steps=3000, decay_rate=0.85,
                  adaptive_fraction=0.3, n_hard_pool=4000,
                  n_candidate_pool=20000, refine_every=500,
                  sampling="adaptive"):
    """Train with the EXACT-Phi distance-residual loss (eps = 0).

    seed        controls the NETWORK INITIALISATION,
    sample_seed controls the COLLOCATION SAMPLING stream (defaults to
                seed + 1000); the two are deliberately independent
                (referee request: separate init and sampling seeds).
    sampling    'adaptive'  30% of the batch drawn from a pool of points
                            with smallest |u_theta| (state-based, as in the
                            original submission),
                'uniform'   100% uniform sampling (ablation),
                'rar'       30% of the batch drawn from a pool of points
                            with LARGEST distance-residual (residual-based
                            adaptive refinement, ablation).
    """
    if sample_seed is None:
        sample_seed = seed + 1000
    tf.random.set_seed(seed)
    np.random.seed(seed)
    net = build_mlp(in_dim=3, hidden=hidden, n_hidden_layers=n_hidden_layers)
    # From here on the global TF stream drives sampling only (the network
    # weights are already initialised), so re-seed it independently:
    tf.random.set_seed(sample_seed)
    schedule = HoldThenExponentialDecay(lr, hold_steps, decay_steps,
                                        decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    if sampling == "uniform":
        adaptive_fraction = 0.0

    # Fixed validation set, IDENTICAL across all runs/seeds/strategies
    # (also reused by the eps-sensitivity check, F2)
    rng_val = np.random.default_rng(20261777)
    t_val = tf.constant(rng_val.uniform(0.0, T, (n_val, 1)), tf.float32)
    x_val = tf.constant(rng_val.uniform(0.0, 1.0, (n_val, 1)), tf.float32)
    y_val = tf.constant(rng_val.uniform(0.0, 1.0, (n_val, 1)), tf.float32)

    n_adapt = int(round(adaptive_fraction * n_colloc))
    n_unif = n_colloc - n_adapt
    hard_pool = None

    @tf.function
    def train_step(t, x, y):
        with tf.GradientTape() as tape:
            u, z = pde_operator(net, t, x, y)
            loss = tf.reduce_mean(relay_distance_squared(z, u, lam, eps=0.0))
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss

    def val_loss_fn():
        u, z = pde_operator(net, t_val, x_val, y_val)
        return float(tf.reduce_mean(
            relay_distance_squared(z, u, lam, eps=0.0)))

    def refresh_hard_pool():
        tc = tf.random.uniform((n_candidate_pool, 1), 0.0, T)
        xc = tf.random.uniform((n_candidate_pool, 1), 0.0, 1.0)
        yc = tf.random.uniform((n_candidate_pool, 1), 0.0, 1.0)
        if sampling == "rar":
            # residual-based adaptive refinement: keep LARGEST residuals
            uc, zc = pde_operator(net, tc, xc, yc)
            score = relay_distance_squared(zc, uc, lam, eps=0.0)[:, 0]
            idx = tf.argsort(score, direction="DESCENDING")[:n_hard_pool]
        else:
            # state-based (original): keep points closest to the free
            # boundary |u_theta| = 0
            uc = u_theta(net, tc, xc, yc)
            idx = tf.argsort(tf.abs(uc[:, 0]))[:n_hard_pool]
        return (tf.gather(tc, idx), tf.gather(xc, idx), tf.gather(yc, idx))

    loss_history, val_history = [], []
    for epoch in range(1, n_epochs + 1):
        if n_adapt > 0 and (epoch == 1 or epoch % refine_every == 0):
            hard_pool = refresh_hard_pool()
        t_u = tf.random.uniform((n_unif, 1), 0.0, T)
        x_u = tf.random.uniform((n_unif, 1), 0.0, 1.0)
        y_u = tf.random.uniform((n_unif, 1), 0.0, 1.0)
        if n_adapt > 0:
            sel = tf.random.uniform((n_adapt,), 0, hard_pool[0].shape[0],
                                    dtype=tf.int32)
            t_b = tf.concat([t_u, tf.gather(hard_pool[0], sel)], axis=0)
            x_b = tf.concat([x_u, tf.gather(hard_pool[1], sel)], axis=0)
            y_b = tf.concat([y_u, tf.gather(hard_pool[2], sel)], axis=0)
        else:
            t_b, x_b, y_b = t_u, x_u, y_u
        loss = train_step(t_b, x_b, y_b)
        loss_history.append(float(loss))
        if epoch % val_every == 0:
            val_history.append((epoch, val_loss_fn()))
        if epoch == 1 or epoch % log_every == 0:
            print(f"  epoch {epoch:6d} | loss {float(loss):.4e}")
    # Always record a final validation evaluation at the last epoch, so the
    # reported "final validation loss" is well defined for any epoch budget.
    if not val_history or val_history[-1][0] != n_epochs:
        val_history.append((n_epochs, val_loss_fn()))
    return net, loss_history, val_history, (t_val, x_val, y_val)


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------


def evaluate_network_on_grid(net, T, N=REF_N, n_times=EVAL_N_TIMES):
    _, h = build_laplacian(N)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    times = np.linspace(0.0, T, n_times)
    Xf = tf.constant(X.flatten().reshape(-1, 1), dtype=tf.float32)
    Yf = tf.constant(Y.flatten().reshape(-1, 1), dtype=tf.float32)
    l2norms, snapshots = [], []
    for t_val in times:
        Tf = tf.fill(tf.shape(Xf), tf.constant(float(t_val), tf.float32))
        u_pred = u_theta(net, Tf, Xf, Yf).numpy().reshape(N, N)
        snapshots.append(u_pred)
        l2norms.append(np.sqrt(np.sum(u_pred**2) * h**2))
    return times, np.array(l2norms), np.array(snapshots), X, Y


def detect_extinction(times, l2norms, tol=EXTINCTION_TOL):
    below = np.where(l2norms < tol)[0]
    return float(times[below[0]]) if len(below) > 0 else None


def eps_sensitivity(net, lam, val_points, eps_grid=(0.0, 1e-8, 1e-6, 1e-4)):
    """(F2) Residual of the trained network w.r.t. Phi_eps on the fixed
    validation set, for each eps.  Returns dict eps -> residual and the max
    relative deviation from the eps = 0 value."""
    t_val, x_val, y_val = val_points
    u, z = pde_operator(net, t_val, x_val, y_val)
    out = {}
    for eps in eps_grid:
        out[eps] = float(tf.reduce_mean(
            relay_distance_squared(z, u, lam, eps=eps)))
    base = out[0.0]
    max_rel = max(abs(out[e] - base) / max(base, 1e-30) for e in eps_grid)
    return out, float(max_rel)


# ============================================================================
# Staged pipeline (referee revision):
#
#   --stage reference   reference solver + grid/time-step CONVERGENCE study
#   --stage train       ONE training run:  --lam L --net-seed S
#                       --sampling {adaptive,uniform,rar}
#                       (launched in parallel over GPUs by
#                        scripts/gpu_launcher.py; resumable, one dir per run)
#   --stage analyze     postprocessing of ALL runs from their checkpoints:
#                       threshold-crossing metrics t_first / t_persistent,
#                       rebound, plateau stats, residual distributions on an
#                       independent 100k-point set, multi-seed aggregation
#   --stage tables      LaTeX macros + ready-made tabulars
#   --stage figures     all figures (seconds, from analyze outputs)
#
# Terminology (referee): the network never reaches exact zero, so the
# quantity extracted from a threshold is reported as a THRESHOLD-CROSSING
# time, not an extinction time.  For each epsilon we report
#   t_first(eps)      = min{t : ||u(t)|| < eps},
#   t_persistent(eps) = min{t : ||u(s)|| < eps for all s in [t, T]},
#   r_post            = max_{t >= t*_ref} ||u(t)||.
# ============================================================================

EXP = "exp63"
EPS_GRID = [5e-4, 1e-3, 2e-3, 5e-3]
REFINEMENT_CONFIGS = [(49, 5e-4), (49, 2.5e-4), (97, 5e-4), (97, 2.5e-4)]
RESIDUAL_FRONT_THRESHOLD = 0.01     # |u_theta| < 0.01  <=>  "near the front"


def _dt_tag(dt):
    return f"{dt:.1e}".replace("-0", "-").replace("e-", "em")


def runs_root():
    d = os.path.join(R.raw_dir(EXP), "runs")
    os.makedirs(d, exist_ok=True)
    return d


def run_tag(lam, net_seed, sampling):
    return f"lam{lam}_s{net_seed}_{sampling}"


def run_dir_for(tag):
    d = os.path.join(runs_root(), tag)
    os.makedirs(d, exist_ok=True)
    return d


def job_grid(params):
    """The pre-registered training grid.  Seeds are FIXED IN THE CONFIG
    (configs/exp63.json) before looking at any results, with the network
    initialisation seed and the sampling seed decoupled inside
    train_dr_pinn (sample_seed = net_seed + 1000)."""
    jobs = []
    for lam in params["lambdas"]:
        for seed in params["seeds"]:
            jobs.append(dict(lam=lam, net_seed=seed, sampling="adaptive"))
    for sampling in params["ablation_samplings"]:
        jobs.append(dict(lam=params["ablation_lambda"],
                         net_seed=params["ablation_seed"],
                         sampling=sampling))
    return jobs


# ----------------------------------------------------------------------------
# Stage: REFERENCE -- baseline curves + a-posteriori convergence check of
# the reference solver (grid refinement 49 -> 97, time step 5e-4 -> 2.5e-4).
# CPU only, minutes.  (Referee item 1.5.)
# ----------------------------------------------------------------------------


def _interp_to_coarse(u_fine, N_fine, N_coarse):
    from scipy.interpolate import RegularGridInterpolator
    hf = 1.0 / (N_fine + 1)
    hc = 1.0 / (N_coarse + 1)
    xf = np.linspace(hf, 1 - hf, N_fine)
    xc = np.linspace(hc, 1 - hc, N_coarse)
    itp = RegularGridInterpolator((xf, xf), u_fine, bounds_error=False,
                                  fill_value=None)
    Xc, Yc = np.meshgrid(xc, xc, indexing="ij")
    return itp(np.stack([Xc.ravel(), Yc.ravel()], axis=1)).reshape(
        N_coarse, N_coarse)


def stage_reference(params):
    t0 = time.time()
    raw = {}
    man = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "baseline_config": {"N": REF_N, "dt": REF_DT},
           "refinement_configs": [list(c) for c in REFINEMENT_CONFIGS],
           "curves": {}, "refinement": {}}

    # Baseline curves for all lambdas (incl. the lambda = 0 control case)
    for lam in LAMBDA_SWEEP_REF:
        times, norms, u_fin, text, X, Y = run_reference_solver(lam)
        raw[f"ref_times_{lam}"] = times
        raw[f"ref_norms_{lam}"] = norms
        raw[f"ref_text_{lam}"] = np.float64(np.nan if text is None else text)
        man["curves"][str(lam)] = {"t_ext": text}
        st = f"t* = {text:.4f}" if text is not None else "not extinguished"
        print(f"  baseline lambda = {lam}: {st}")
    raw["ref_ufinal_baseline"] = run_reference_solver(LAMBDA_BASELINE)[2]

    # Convergence study for the main sweep (referee item 1.5): report the
    # change of t*, of the L2 curve, and of the profile just before
    # extinction, under grid and time-step refinement.
    for lam in LAMBDA_SWEEP_MAIN:
        base = None
        man["refinement"][str(lam)] = {}
        for (N, dt) in REFINEMENT_CONFIGS:
            times, norms, u_fin, text, _, _ = run_reference_solver(
                lam, N=N, dt=dt)
            key = f"N{N}_dt{_dt_tag(dt)}"
            raw[f"refine_times_{lam}_{key}"] = times
            raw[f"refine_norms_{lam}_{key}"] = norms
            t_pre = 0.9 * text if text is not None else 0.9 * T_HORIZON
            # profile shortly before extinction, on this config's grid
            u_pre = _profile_at_time(lam, N, dt, t_pre)
            entry = {"N": N, "dt": dt, "t_ext": text}
            if base is None:
                base = {"times": times, "norms": norms, "t_ext": text,
                        "u_pre": u_pre, "N": N}
                entry.update(dict(dt_ext_vs_base=0.0, curve_maxdiff=0.0,
                                  profile_maxdiff=0.0))
            else:
                nb = np.interp(base["times"], times, norms)
                u_cmp = (u_pre if N == base["N"]
                         else _interp_to_coarse(u_pre, N, base["N"]))
                entry.update(dict(
                    dt_ext_vs_base=(None if text is None or
                                    base["t_ext"] is None
                                    else text - base["t_ext"]),
                    curve_maxdiff=float(np.max(np.abs(nb - base["norms"]))),
                    profile_maxdiff=float(np.max(np.abs(u_cmp
                                                        - base["u_pre"])))))
            man["refinement"][str(lam)][key] = entry
            print(f"  refine lambda={lam} N={N} dt={dt:g}: t*={text}, "
                  f"d(t*)={entry['dt_ext_vs_base']}, "
                  f"curve maxdiff={entry['curve_maxdiff']:.2e}")

    np.savez_compressed(os.path.join(R.raw_dir(EXP), "reference.npz"), **raw)
    with open(os.path.join(R.raw_dir(EXP), "reference_manifest.json"),
              "w") as f:
        json.dump(man, f, indent=2)
    print(f"[reference] done in {time.time() - t0:.0f} s.")


def _profile_at_time(lam, N, dt, t_stop):
    """Reference profile u(t_stop) for one solver configuration."""
    Lap, h = build_laplacian(N)
    x = np.linspace(h, 1 - h, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Asys = (sp.identity(N * N, format="csc") - dt * Lap).tocsc()
    solve = spla.factorized(Asys)
    u = u0_numpy(X, Y).flatten()
    n_steps = int(round(t_stop / dt))
    for _ in range(n_steps):
        u = soft_threshold(solve(u), dt * lam)
    return u.reshape(N, N)


def load_reference():
    path = os.path.join(R.raw_dir(EXP), "reference.npz")
    manp = os.path.join(R.raw_dir(EXP), "reference_manifest.json")
    if not (os.path.exists(path) and os.path.exists(manp)):
        raise FileNotFoundError(
            "reference outputs not found -- run "
            "`python3 experiments/run_experiment_63.py --stage reference`")
    with open(manp) as f:
        man = json.load(f)
    return np.load(path), man


# ----------------------------------------------------------------------------
# Stage: TRAIN -- ONE run (one lambda, one seed, one sampling strategy).
# Designed to be launched many times in parallel by scripts/gpu_launcher.py
# (one process per GPU).  Writes results/raw/exp63/runs/<tag>/.
# ----------------------------------------------------------------------------


def _setup_gpu():
    """Never grab the whole card: the 4x10GB machine is shared."""
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def stage_train_single(params, lam, net_seed, sampling):
    _setup_gpu()
    tag = run_tag(lam, net_seed, sampling)
    rdir = run_dir_for(tag)
    done_marker = os.path.join(rdir, "run.json")
    if os.path.exists(done_marker):
        with open(done_marker) as f:
            if json.load(f).get("status") == "done":
                print(f"[train {tag}] already done -- skipping.")
                return

    n_epochs = int(params["n_epochs"])
    print(f"[train {tag}] n_epochs={n_epochs}, "
          f"net_seed={net_seed}, sample_seed={net_seed + 1000}, "
          f"sampling={sampling}")
    t0 = time.time()
    net, loss_hist, val_hist, _ = train_dr_pinn(
        lam=lam, T=T_HORIZON, n_epochs=n_epochs,
        seed=net_seed, sample_seed=net_seed + 1000, sampling=sampling)
    train_seconds = time.time() - t0
    net.save_weights(os.path.join(rdir, "net.weights.h5"))

    times, l2, snaps, _, _ = evaluate_network_on_grid(net, T_HORIZON)
    val_loss = val_hist[-1][1] if val_hist else float("nan")

    np.savez_compressed(
        os.path.join(rdir, "run.npz"),
        loss_hist=np.asarray(loss_hist, dtype=np.float64),
        val_epochs=np.asarray([e for e, _ in val_hist]),
        val_vals=np.asarray([v for _, v in val_hist]),
        times=times, l2=l2, snap_final=snaps[-1])
    with open(done_marker, "w") as f:
        json.dump({
            "status": "done", "tag": tag, "lam": lam,
            "net_seed": net_seed, "sample_seed": net_seed + 1000,
            "sampling": sampling, "n_epochs": n_epochs,
            "val_loss": float(val_loss),
            "train_seconds": train_seconds,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": R.hardware_string(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES",
                                                   "<unset>"),
            "versions": {"python": platform.python_version(),
                         "numpy": np.__version__,
                         "tensorflow": tf.__version__},
        }, f, indent=2)
    print(f"[train {tag}] done in {train_seconds:.0f} s, "
          f"val loss = {val_loss:.4e}")


# ----------------------------------------------------------------------------
# Stage: ANALYZE -- postprocessing of every finished run FROM ITS CHECKPOINT.
# No retraining.  (Referee items 1.1, 1.2, 1.4, 1.6, 2.)
# ----------------------------------------------------------------------------


def threshold_metrics(times, l2, eps, text_ref):
    """t_first, t_persistent, rebound after first crossing; plus min-norm
    and post-t*_ref statistics (eps-independent)."""
    below = l2 < eps
    idx = np.where(below)[0]
    t_first = float(times[idx[0]]) if len(idx) else None
    # persistent: from the first index such that all later values are below
    t_persistent = None
    if below[-1]:
        k = len(below) - 1
        while k >= 0 and below[k]:
            k -= 1
        t_persistent = float(times[k + 1])
    rebound = float(np.max(l2[idx[0]:])) if len(idx) else None
    out = {"eps": eps, "t_first": t_first, "t_persistent": t_persistent,
           "rebound_after_first_crossing": rebound}
    return out


def norm_curve_stats(times, l2, text_ref):
    stats = {"min_norm": float(np.min(l2)),
             "t_argmin": float(times[int(np.argmin(l2))])}
    if text_ref is not None:
        m = times >= text_ref
        if m.any():
            post = l2[m]
            stats["post_ref_max"] = float(np.max(post))       # r_post
            stats["post_ref_median"] = float(np.median(post))
            stats["post_ref_min"] = float(np.min(post))
    return stats


def residual_statistics(net, lam, n_points, batch, seed=20260999):
    """Pointwise distance residual r = dist(z, Phi(s)) on an INDEPENDENT
    freshly sampled set (identical across runs), split near/away from the
    free boundary |u_theta| < 0.01.  (Referee item 1.4.)"""
    rng = np.random.default_rng(seed)
    t_np = rng.uniform(0.0, T_HORIZON, (n_points, 1)).astype(np.float32)
    x_np = rng.uniform(0.0, 1.0, (n_points, 1)).astype(np.float32)
    y_np = rng.uniform(0.0, 1.0, (n_points, 1)).astype(np.float32)
    rs, us = [], []
    for k in range(0, n_points, batch):
        tb = tf.constant(t_np[k:k + batch])
        xb = tf.constant(x_np[k:k + batch])
        yb = tf.constant(y_np[k:k + batch])
        u, z = pde_operator(net, tb, xb, yb)
        d2 = relay_distance_squared(z, u, lam, eps=0.0)
        rs.append(np.sqrt(np.maximum(d2.numpy()[:, 0], 0.0)))
        us.append(u.numpy()[:, 0])
    r = np.concatenate(rs)
    u = np.concatenate(us)
    front = np.abs(u) < RESIDUAL_FRONT_THRESHOLD

    def _s(v):
        if len(v) == 0:
            return {"n": 0}
        return {"n": int(len(v)), "mean": float(np.mean(v)),
                "rms": float(np.sqrt(np.mean(v ** 2))),
                "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)),
                "p95": float(np.percentile(v, 95)),
                "p99": float(np.percentile(v, 99)),
                "max": float(np.max(v))}

    stats = {"global": _s(r), "front": _s(r[front]),
             "away": _s(r[~front]),
             "front_threshold": RESIDUAL_FRONT_THRESHOLD,
             "front_fraction": float(np.mean(front))}
    return stats, r, front


def _load_net_from_run(rdir):
    net = build_mlp()
    # build variables with a dummy forward pass, then load
    z = tf.zeros((1, 1), tf.float32)
    u_theta(net, z, z, z)
    net.load_weights(os.path.join(rdir, "net.weights.h5"))
    return net


def _agg(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": float(np.median(v)),
            "min": float(np.min(v)), "max": float(np.max(v))}


def stage_analyze(params):
    _setup_gpu()
    t0 = time.time()
    ref_raw, ref_man = load_reference()
    n_res = int(params["n_residual_points"])
    res_batch = int(params["residual_batch"])

    run_jsons = sorted(glob.glob(os.path.join(runs_root(), "*", "run.json")))
    runs = []
    for rj in run_jsons:
        with open(rj) as f:
            meta = json.load(f)
        if meta.get("status") == "done":
            runs.append(meta)
    if not runs:
        raise FileNotFoundError(
            "no finished runs under results/raw/exp63/runs/ -- launch the "
            "training grid first (python3 scripts/gpu_launcher.py)")
    print(f"[analyze] {len(runs)} finished runs found")

    fig_raw = {}   # arrays for the figures stage
    per_run = {}
    for meta in runs:
        tag = meta["tag"]
        lam = meta["lam"]
        rdir = run_dir_for(tag)
        d = np.load(os.path.join(rdir, "run.npz"))
        text_ref = ref_man["curves"][str(lam)]["t_ext"]
        times, l2 = d["times"], d["l2"]

        thr = {f"{eps:g}": threshold_metrics(times, l2, eps, text_ref)
               for eps in EPS_GRID}
        curve = norm_curve_stats(times, l2, text_ref)

        print(f"  {tag}: residual stats on {n_res} fresh points ...")
        net = _load_net_from_run(rdir)
        res, r, front = residual_statistics(net, lam, n_res, res_batch)

        analysis = {"tag": tag, "lam": lam, "sampling": meta["sampling"],
                    "net_seed": meta["net_seed"],
                    "val_loss": meta["val_loss"],
                    "train_seconds": meta.get("train_seconds"),
                    "t_ext_ref": text_ref,
                    "thresholds": thr, "norm_curve": curve,
                    "residuals": res}
        with open(os.path.join(rdir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        per_run[tag] = analysis

        fig_raw[f"curve_times_{tag}"] = times
        fig_raw[f"curve_l2_{tag}"] = l2
        rng_sub = np.random.default_rng(7)
        for name, mask in (("front", front), ("away", ~front)):
            v = r[mask]
            if len(v) > 5000:
                v = rng_sub.choice(v, 5000, replace=False)
            fig_raw[f"res_{name}_{tag}"] = v

    # ---- multi-seed aggregation per (lambda, sampling) --------------------
    def sel(lam, sampling):
        return [a for a in per_run.values()
                if a["lam"] == lam and a["sampling"] == sampling]

    aggregated = {}
    for lam in params["lambdas"]:
        rs = sel(lam, "adaptive")
        entry = {"n_runs": len(rs),
                 "seeds": sorted(a["net_seed"] for a in rs),
                 "n_failed": 0,
                 "val_loss": _agg([a["val_loss"] for a in rs]),
                 "rms_global": _agg([a["residuals"]["global"].get("rms")
                                     for a in rs]),
                 "post_ref_max": _agg([a["norm_curve"].get("post_ref_max")
                                       for a in rs]),
                 "min_norm": _agg([a["norm_curve"]["min_norm"]
                                   for a in rs])}
        for eps in EPS_GRID:
            k = f"{eps:g}"
            entry[f"t_first_{k}"] = _agg(
                [a["thresholds"][k]["t_first"] for a in rs])
            entry[f"t_persistent_{k}"] = _agg(
                [a["thresholds"][k]["t_persistent"] for a in rs])
            entry[f"n_no_persistent_{k}"] = sum(
                1 for a in rs if a["thresholds"][k]["t_persistent"] is None)
        aggregated[str(lam)] = entry

    # ---- diagnostics tied to ONE representative checkpoint ----------------
    # median-val-loss adaptive run for the baseline lambda; the branch
    # diagnostic and the eps-sensitivity check are recomputed from its
    # saved weights (no retraining).
    base_runs = sorted(sel(LAMBDA_BASELINE, "adaptive"),
                       key=lambda a: a["val_loss"])
    rep = base_runs[len(base_runs) // 2] if base_runs else None
    eps_sens = None
    if rep is not None:
        rep_dir = run_dir_for(rep["tag"])
        net = _load_net_from_run(rep_dir)
        tf.random.set_seed(1)
        n_diag = 4000
        t_d = tf.random.uniform((n_diag, 1), 0.0, T_HORIZON)
        x_d = tf.random.uniform((n_diag, 1), 0.0, 1.0)
        y_d = tf.random.uniform((n_diag, 1), 0.0, 1.0)
        u_d, z_d = pde_operator(net, t_d, x_d, y_d)
        fig_raw["diag_s_vals"] = u_d.numpy().flatten()
        fig_raw["diag_z_vals"] = z_d.numpy().flatten()
        fig_raw["rep_snap_final"] = np.load(
            os.path.join(rep_dir, "run.npz"))["snap_final"]
        rng_val = np.random.default_rng(20261777)
        val_pts = tuple(tf.constant(rng_val.uniform(lo, hi, (4000, 1)),
                                    tf.float32)
                        for lo, hi in ((0.0, T_HORIZON), (0.0, 1.0),
                                       (0.0, 1.0)))
        vals, max_rel = eps_sensitivity(net, LAMBDA_BASELINE, val_pts)
        eps_sens = {"values": {f"{k:g}": v for k, v in vals.items()},
                    "max_rel_dev": max_rel}
        # legacy single-run diagnostics for the representative checkpoint
        # (kept so the current manuscript still compiles; the multi-seed
        # aggregates supersede them in the revised text)
        rep_npz = np.load(os.path.join(rep_dir, "run.npz"))
        lh = rep_npz["loss_hist"]
        plateau_window = lh[100:min(4000, len(lh))]
        plateau_level = (float(np.median(plateau_window))
                         if len(plateau_window) else 0.0)
        escape_epoch = next(
            (int(i) for i, v in enumerate(lh)
             if plateau_level > 0 and i > 100 and v < plateau_level / 10),
            None)
        tr = ref_raw[f"ref_times_{LAMBDA_BASELINE}"]
        nr = ref_raw[f"ref_norms_{LAMBDA_BASELINE}"]
        l2i = np.interp(rep_npz["times"], tr, nr)
        rep_metrics = {
            "plateau_level": plateau_level,
            "escape_epoch": escape_epoch,
            "norm_disc": float(np.max(np.abs(rep_npz["l2"] - l2i))
                               / (l2i.max() + 1e-12)),
            "max_pt_err": float(np.abs(rep_npz["snap_final"]
                                       - ref_raw["ref_ufinal_baseline"])
                                .max()),
            "z_front_max": float(np.max(np.abs(fig_raw["diag_z_vals"]))),
        }

    manifest = {
        "script": "run_experiment_63.py --stage analyze",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - t0,
        "n_epochs": int(params["n_epochs"]),
        "smoke": bool(params.get("_smoke", False)),
        "hardware": R.hardware_string(),
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__,
                     "tensorflow": tf.__version__},
        "eps_grid": EPS_GRID,
        "n_residual_points": n_res,
        "extinction_tolerance_paper": EXTINCTION_TOL,
        "eval_time_step": T_HORIZON / (EVAL_N_TIMES - 1),
        "reference": ref_man,
        "representative_run": rep["tag"] if rep else None,
        "representative_metrics": rep_metrics if rep is not None else None,
        "eps_sensitivity": eps_sens,
        "runs": per_run,
        "aggregated": aggregated,
    }
    R.save_manifest(EXP, manifest, "63")
    np.savez_compressed(os.path.join(R.raw_dir(EXP), "analysis.npz"),
                        **{k: np.asarray(v) for k, v in fig_raw.items()})
    print(f"[analyze] done in {time.time() - t0:.0f} s.")


# ----------------------------------------------------------------------------
# Stage: TABLES -- LaTeX macros + ready-made tabulars in results/aggregated/.
# Honest terminology: "threshold-crossing time", never "extinction time".
# ----------------------------------------------------------------------------

NO_PERSISTENT = r"no persistent ext.\ before $T$"


def _fmt_med_range(agg, digits=4, none_text="--"):
    if agg.get("n", 0) == 0:
        return none_text
    if agg["n"] == 1:
        return f"{agg['median']:.{digits}f}"
    return (f"{agg['median']:.{digits}f} "
            f"[{agg['min']:.{digits}f}, {agg['max']:.{digits}f}]")


def stage_tables(params):
    man = R.load_manifest(EXP, "63")
    agg = man["aggregated"]
    runs = man["runs"]
    ref = man["reference"]
    out = R.aggregated_dir()

    # ---- macros ------------------------------------------------------------
    macros = {}
    for lam, key in zip(params["lambdas"], ["A", "B", "C"]):
        a = agg[str(lam)]
        macros[f"RelayTRef{key}"] = \
            f"{ref['curves'][str(lam)]['t_ext']:.4f}"
        macros[f"RelayTFirst{key}"] = _fmt_med_range(a["t_first_0.001"])
        tp = a["t_persistent_0.001"]
        macros[f"RelayTPersist{key}"] = (
            _fmt_med_range(tp) if tp.get("n", 0) > 0 else NO_PERSISTENT)
        macros[f"RelayNoPersist{key}"] = str(a["n_no_persistent_0.001"])
        macros[f"RelayRPost{key}"] = (
            R.sci_tex(a["post_ref_max"]["median"])
            if a["post_ref_max"].get("n") else r"\text{--}")
        macros[f"RelayValLoss{key}"] = (
            R.sci_tex(a["val_loss"]["median"])
            if a["val_loss"].get("n") else r"\text{--}")
        macros[f"RelayNSeeds{key}"] = str(a["n_runs"])
    macros["RelaySeedList"] = ", ".join(
        str(s) for s in agg[str(params['lambdas'][0])]["seeds"])
    macros["RelayEpochs"] = f"{man['n_epochs']:,}".replace(",", r"\,")
    macros["RelayNResPoints"] = \
        f"{man['n_residual_points']:,}".replace(",", r"\,")
    if man.get("eps_sensitivity"):
        macros["RelayEpsSens"] = R.sci_tex(
            man["eps_sensitivity"]["max_rel_dev"])
    if man.get("representative_run"):
        macros["RelayRepRun"] = man["representative_run"].replace("_", r"\_")

    # ---- legacy aliases so the CURRENT manuscript keeps compiling ----------
    # (values are the HONEST multi-seed / representative-run quantities;
    # the revised text should switch to the new macro names above)
    rm = man.get("representative_metrics") or {}
    base = agg[str(LAMBDA_BASELINE)]
    if base["val_loss"].get("n"):
        macros["RelayValLoss"] = R.sci_tex(base["val_loss"]["median"])
        macros["RelayRMS"] = f"{np.sqrt(base['val_loss']['median']):.2f}"
    if rm:
        macros["RelayPlateauLevel"] = f"{rm['plateau_level']:.0f}"
        macros["RelayEscapeEpoch"] = (
            f"{rm['escape_epoch']:,}".replace(",", r"\,")
            if rm.get("escape_epoch") else "--")
        macros["RelayNormDisc"] = f"{rm['norm_disc'] * 100:.2f}" + r"\%"
        macros["RelayMaxPtErr"] = R.sci_tex(rm["max_pt_err"])
        macros["RelayZFrontMax"] = f"{rm['z_front_max']:.0f}"
    if base["post_ref_max"].get("n"):
        macros["RelayPostExtPlateau"] = (
            r"\approx" + R.sci_tex(base["post_ref_max"]["median"], 0))
    macros["RelayTRefBase"] =         f"{ref['curves'][str(LAMBDA_BASELINE)]['t_ext']:.4f}"
    abs_errs = []
    for lam, key in zip(params["lambdas"], ["A", "B", "C"]):
        a = agg[str(lam)]["t_first_0.001"]
        t_ext = ref["curves"][str(lam)]["t_ext"]
        if a.get("n") and t_ext:
            macros[f"RelayTNet{key}"] = f"{a['median']:.4f}"
            rel = abs(a["median"] - t_ext) / t_ext
            macros[f"RelayRelErr{key}"] = f"${rel * 100:.1f}" + r"\%$"
            abs_errs.append(abs(a["median"] - t_ext))
        else:
            macros[f"RelayTNet{key}"] = "--"
            macros[f"RelayRelErr{key}"] = "--"
    if abs_errs:
        macros["RelayAbsErrMin"] = f"{min(abs_errs):.3f}"
        macros["RelayAbsErrMax"] = f"{max(abs_errs):.3f}"

    # solver convergence deltas (worst over lambdas, finest vs baseline)
    fin_key = f"N97_dt{_dt_tag(2.5e-4)}"
    dts, curves = [], []
    for lam in params["lambdas"]:
        e = ref["refinement"][str(lam)].get(fin_key)
        if e:
            if e["dt_ext_vs_base"] is not None:
                dts.append(abs(e["dt_ext_vs_base"]))
            curves.append(e["curve_maxdiff"])
    if dts:
        macros["RelaySolverDtExtMax"] = f"{max(dts):.4f}"
    if curves:
        macros["RelaySolverCurveMax"] = R.sci_tex(max(curves))

    # ablation macros (same seed, lambda = ablation_lambda)
    ab_lam = params["ablation_lambda"]
    ab_seed = params["ablation_seed"]
    for sampling, key in (("adaptive", "Adap"), ("uniform", "Unif"),
                          ("rar", "Rar")):
        a = runs.get(run_tag(ab_lam, ab_seed, sampling))
        if a is None:
            continue
        macros[f"RelayAbl{key}ValLoss"] = R.sci_tex(a["val_loss"])
        macros[f"RelayAbl{key}RMSFront"] = R.sci_tex(
            a["residuals"]["front"].get("rms"))
        tf_ = a["thresholds"]["0.001"]["t_first"]
        macros[f"RelayAbl{key}TFirst"] = (f"{tf_:.4f}" if tf_ else "--")

    R.write_macros(os.path.join(out, "results_63.tex"), macros,
                   "run_experiment_63.py --stage tables")

    # ---- ready-made tabulars ----------------------------------------------
    def tab(fname, header, rows, caption_comment):
        with open(os.path.join(out, fname), "w") as f:
            f.write(f"% AUTO-GENERATED -- {caption_comment}\n")
            f.write(r"\begin{tabular}{" + header[0] + "}\n\\toprule\n")
            f.write(header[1] + r" \\" + "\n\\midrule\n")
            for r_ in rows:
                f.write(r_ + r" \\" + "\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"Wrote {os.path.join(out, fname)}")

    # Table: threshold sensitivity (per lambda, adaptive runs, median[range])
    rows = []
    for lam in params["lambdas"]:
        a = agg[str(lam)]
        for eps in EPS_GRID:
            k = f"{eps:g}"
            tp = a[f"t_persistent_{k}"]
            rows.append(
                rf"${lam}$ & ${R.sci_tex(eps, 0)}$ & "
                f"{_fmt_med_range(a[f't_first_{k}'])} & "
                + (f"{_fmt_med_range(tp)}" if tp.get("n", 0) > 0
                   else NO_PERSISTENT)
                + f" & {a[f'n_no_persistent_{k}']}/{a['n_runs']}")
    tab("table_63_thresholds.tex",
        ("llllc",
         r"$\lambda$ & $\varepsilon$ & $t_{\mathrm{first}}$ & "
         r"$t_{\mathrm{persistent}}$ & no pers."),
        rows, "threshold sensitivity (referee item 1.2)")

    # Table: residual distribution per lambda (median run per lambda)
    rows = []
    for lam in params["lambdas"]:
        rs = sorted([a for a in runs.values()
                     if a["lam"] == lam and a["sampling"] == "adaptive"],
                    key=lambda a: a["val_loss"])
        if not rs:
            continue
        a = rs[len(rs) // 2]["residuals"]
        rows.append(
            rf"${lam}$ & ${R.sci_tex(a['global']['rms'])}$ & "
            rf"${R.sci_tex(a['global']['p95'])}$ & "
            rf"${R.sci_tex(a['global']['max'])}$ & "
            rf"${R.sci_tex(a['front']['rms'])}$ & "
            rf"${R.sci_tex(a['away']['rms'])}$")
    tab("table_63_residuals.tex",
        ("lccccc",
         r"$\lambda$ & RMS & $p_{95}$ & max & RMS front & RMS away"),
        rows, "residual distribution on independent points "
              "(referee item 1.4; front = $|u_\\theta|<0.01$)")

    # Table: per-run (seed) results
    rows = []
    for a in sorted(runs.values(),
                    key=lambda a: (a["lam"], a["sampling"], a["net_seed"])):
        t1 = a["thresholds"]["0.001"]
        rows.append(
            rf"${a['lam']}$ & {a['sampling']} & {a['net_seed']} & "
            rf"${R.sci_tex(a['val_loss'])}$ & "
            + (f"{t1['t_first']:.4f}" if t1["t_first"] else "--") + " & "
            + (f"{t1['t_persistent']:.4f}" if t1["t_persistent"]
               else NO_PERSISTENT) + " & "
            + (f"${R.sci_tex(a['norm_curve'].get('post_ref_max'))}$"
               if a["norm_curve"].get("post_ref_max") is not None else "--"))
    tab("table_63_runs.tex",
        ("llccccc",
         r"$\lambda$ & sampling & seed & val.\ loss & "
         r"$t_{\mathrm{first}}(10^{-3})$ & $t_{\mathrm{persistent}}$ & "
         r"$r_{\mathrm{post}}$"),
        rows, "per-run results (referee item 2)")

    # Table: reference-solver convergence
    rows = []
    for lam in params["lambdas"]:
        for key, e in ref["refinement"][str(lam)].items():
            rows.append(
                rf"${lam}$ & ${e['N']}\times{e['N']}$ & ${e['dt']:g}$ & "
                + (f"{e['t_ext']:.4f}" if e["t_ext"] else "--") + " & "
                + (f"{e['dt_ext_vs_base']:+.4f}"
                   if e["dt_ext_vs_base"] is not None else "--") + " & "
                rf"${R.sci_tex(e['curve_maxdiff'])}$ & "
                rf"${R.sci_tex(e['profile_maxdiff'])}$")
    tab("table_63_solver_convergence.tex",
        ("lcccccc",
         r"$\lambda$ & grid & $\Delta t$ & $t^*$ & $\Delta t^*$ & "
         r"$\max|\Delta\|u\||$ & $\max|\Delta u_{\mathrm{pre}}|$"),
        rows, "reference-solver convergence (referee item 1.5)")

    # Table: plateau statistics after t*_ref (referee item 1.6)
    rows = []
    for a in sorted(runs.values(),
                    key=lambda a: (a["lam"], a["sampling"], a["net_seed"])):
        nc = a["norm_curve"]
        if "post_ref_median" not in nc:
            continue
        rows.append(
            rf"${a['lam']}$ & {a['sampling']} & {a['net_seed']} & "
            rf"${R.sci_tex(nc['post_ref_median'])}$ & "
            rf"${R.sci_tex(nc['post_ref_min'])}$ & "
            rf"${R.sci_tex(nc['post_ref_max'])}$")
    tab("table_63_plateau.tex",
        ("llcccc",
         r"$\lambda$ & sampling & seed & median & min & max"),
        rows, r"$\|u_\theta(t)\|$ for $t \ge t^*_{\mathrm{ref}}$ "
              "(referee item 1.6)")

    with open(os.path.join(out, "manifest_63.json"), "w") as f:
        json.dump(man, f, indent=2)


# ----------------------------------------------------------------------------
# Stage: FIGURES -- from reference.npz + analysis.npz (seconds).
# ----------------------------------------------------------------------------


def _refc(ref_raw, lam):
    text = float(ref_raw[f"ref_text_{lam}"])
    return (ref_raw[f"ref_times_{lam}"], ref_raw[f"ref_norms_{lam}"],
            None if np.isnan(text) else text)


def plot_phi_graph(ax, lam, s_min, s_max, lw=2, label=r"graph of $\Phi$"):
    if s_max > 0:
        ax.plot([max(s_min, 0), s_max], [-lam, -lam], color="tab:red", lw=lw,
                label=label, zorder=3)
        label = None
    if s_min < 0:
        ax.plot([s_min, min(s_max, 0)], [lam, lam], color="tab:red", lw=lw,
                label=label, zorder=3)
        label = None
    ax.plot([0, 0], [-lam, lam], color="tab:red", lw=lw, label=label, zorder=3)


def stage_figures(params):
    ref_raw, ref_man = load_reference()
    ana = np.load(os.path.join(R.raw_dir(EXP), "analysis.npz"))
    man = R.load_manifest(EXP, "63")
    fig_dir = R.figures_dir()
    runs = man["runs"]

    def tags(lam, sampling="adaptive"):
        return sorted(t for t, a in runs.items()
                      if a["lam"] == lam and a["sampling"] == sampling)

    # ---- reference_extinction_curves (unchanged) ---------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for lam in LAMBDA_SWEEP_REF:
        times, norms, text = _refc(ref_raw, lam)
        label = rf"$\lambda={lam}$" + (" (heat eq.)" if lam == 0.0 else "")
        ax.semilogy(times, np.maximum(norms, 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2, label=label)
        if text is not None:
            ax.axvline(text, color=LAMBDA_COLORS[lam], ls=":", lw=0.9,
                       alpha=0.8)
            ax.annotate(rf"$t^*={text:.4f}$", xy=(text, NORM_FLOOR * 4),
                        xytext=(text + 0.004, NORM_FLOOR * 4),
                        fontsize=10, color=LAMBDA_COLORS[lam], rotation=90,
                        va="bottom")
    ax.set_ylim(NORM_FLOOR, 2.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$  (log scale)")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "reference_extinction_curves.png"),
                dpi=180)
    plt.close(fig)

    rep_tag = man.get("representative_run")

    # ---- training_history_relay (representative run) -----------------------
    if rep_tag:
        d = np.load(os.path.join(run_dir_for(rep_tag), "run.npz"))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.semilogy(d["loss_hist"], color="lightsteelblue", lw=0.8,
                    label="training loss (per step)", rasterized=True)
        ax.semilogy(d["val_epochs"], d["val_vals"], color="tab:red", lw=1.6,
                    marker="o", markersize=2.5,
                    label="validation loss (fixed batch)")
        ax.axvline(5000, color="0.4", ls="--", lw=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\mathcal{L}_{\mathrm{incl}}(\theta)$")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(title=rep_tag.replace("_", r"\_"), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "training_history_relay.png"),
                    dpi=180)
        plt.close(fig)

    # ---- dr_pinn_vs_reference (rep run + seed envelope) --------------------
    times_ref, norms_ref, text_ref = _refc(ref_raw, LAMBDA_BASELINE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(times_ref, np.maximum(norms_ref, 1e-16),
                     color="tab:blue", label="Reference solution", lw=2)
    for i, t_ in enumerate(tags(LAMBDA_BASELINE)):
        axes[0].semilogy(ana[f"curve_times_{t_}"],
                         np.maximum(ana[f"curve_l2_{t_}"], 1e-16), "--",
                         color="tab:orange", lw=1.2, alpha=0.9,
                         label="DR-PINN (per seed)" if i == 0 else None)
    if text_ref is not None:
        axes[0].axvline(text_ref, color="gray", ls=":", lw=0.9)
        axes[0].annotate(rf"$t^*_{{\rm ref}}={text_ref:.4f}$",
                         xy=(text_ref, NORM_FLOOR * 4),
                         xytext=(text_ref + 0.006, NORM_FLOOR * 4),
                         rotation=90, va="bottom", fontsize=10, color="0.3")
    axes[0].set_ylim(NORM_FLOOR, 2.0)
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$")
    axes[0].grid(True, which="major", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=9)
    if "rep_snap_final" in ana:
        im = axes[1].imshow(
            np.abs(ana["rep_snap_final"] - ref_raw["ref_ufinal_baseline"]).T,
            origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
        axes[1].set_title(
            rf"$|u_\theta - u_{{\mathrm{{ref}}}}|$ at $t=T={T_HORIZON}$")
        axes[1].set_xlabel("$x$")
        axes[1].set_ylabel("$y$")
        cbar = fig.colorbar(im, ax=axes[1])
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dr_pinn_vs_reference.png"), dpi=180)
    plt.close(fig)

    # ---- branch_selection_diagnostic (with the zoom inset) -----------------
    if "diag_s_vals" in ana:
        s_vals = ana["diag_s_vals"]
        z_vals = ana["diag_z_vals"]
        lam = LAMBDA_BASELINE
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        ax.scatter(s_vals, z_vals, s=5, alpha=0.20, color="tab:blue",
                   edgecolors="none", label="Collocation points",
                   rasterized=True)
        s_min = min(s_vals.min(), -0.1)
        s_max = max(s_vals.max(), 0.1)
        plot_phi_graph(ax, lam, s_min, s_max)
        ax.annotate(rf"$z=-\lambda={-lam}$", xy=(0.55, -lam),
                    xytext=(0.55, -lam - 0.55), fontsize=11, color="tab:red",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
        ax.set_xlabel(r"$s = u_\theta(t,x,y)$")
        ax.set_ylabel(r"$z = \partial_t u_\theta - \Delta u_\theta$")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right")
        axins = ax.inset_axes([0.52, 0.55, 0.44, 0.40])
        axins.scatter(s_vals, z_vals, s=4, alpha=0.25, color="tab:blue",
                      edgecolors="none", rasterized=True)
        plot_phi_graph(axins, lam, -0.02, 0.10, lw=1.6, label=None)
        axins.set_xlim(-0.02, 0.10)
        axins.set_ylim(-1.1, 1.1)
        axins.tick_params(labelsize=9)
        axins.grid(True, alpha=0.2)
        ax.indicate_inset_zoom(axins, edgecolor="0.4")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "branch_selection_diagnostic.png"),
                    dpi=180)
        plt.close(fig)

    # ---- extinction_time_sweep -> threshold-crossing sweep with seeds ------
    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in params["lambdas"]:
        times_r, norms_r, text_r = _refc(ref_raw, lam)
        ax.semilogy(times_r, np.maximum(norms_r, 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2,
                    label=rf"$\lambda={lam}$ (reference)")
        for i, t_ in enumerate(tags(lam)):
            ax.semilogy(ana[f"curve_times_{t_}"],
                        np.maximum(ana[f"curve_l2_{t_}"], 1e-16),
                        color=LAMBDA_COLORS[lam], lw=1.0, ls="--", alpha=0.8,
                        label=(rf"$\lambda={lam}$ (DR-PINN, seeds)"
                               if i == 0 else None))
        if text_r is not None:
            ax.axvline(text_r, color=LAMBDA_COLORS[lam], ls=":", lw=0.9,
                       alpha=0.7)
    ax.axhline(EXTINCTION_TOL, color="0.3", lw=0.8, ls="-.",
               label=rf"$\varepsilon={EXTINCTION_TOL:g}$")
    ax.set_ylim(NORM_FLOOR, 2.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=9, ncol=2, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "extinction_time_sweep.png"), dpi=180)
    plt.close(fig)

    # ---- NEW: residual_distribution (front vs away, per lambda) ------------
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    data, labels, colors = [], [], []
    for lam in params["lambdas"]:
        ts = tags(lam)
        if not ts:
            continue
        rep = sorted(ts, key=lambda t_: runs[t_]["val_loss"])[len(ts) // 2]
        for name, pretty in (("front", "front"), ("away", "away")):
            v = ana[f"res_{name}_{rep}"]
            data.append(np.log10(np.maximum(v, 1e-16)))
            labels.append(rf"$\lambda={lam}$" + f"\n{pretty}")
            colors.append(LAMBDA_COLORS[lam])
    bp = ax.boxplot(data, tick_labels=labels, whis=(5, 99), showfliers=False,
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    ax.set_ylabel(r"$\log_{10}\,\mathrm{dist}(z_\theta,\Phi(u_\theta))$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title(rf"Residual distribution (independent points; "
                 rf"front: $|u_\theta|<{RESIDUAL_FRONT_THRESHOLD}$)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "residual_distribution.png"), dpi=180)
    plt.close(fig)

    # ---- NEW: threshold_sensitivity ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for lam in params["lambdas"]:
        text_r = ref_man["curves"][str(lam)]["t_ext"]
        if text_r is not None:
            ax.axhline(text_r, color=LAMBDA_COLORS[lam], ls=":", lw=1.0)
        for t_ in tags(lam):
            tf_vals = [runs[t_]["thresholds"][f"{e:g}"]["t_first"]
                       for e in EPS_GRID]
            xs = [e for e, v in zip(EPS_GRID, tf_vals) if v is not None]
            ys = [v for v in tf_vals if v is not None]
            ax.semilogx(xs, ys, "o-", color=LAMBDA_COLORS[lam], lw=1.0,
                        ms=4, alpha=0.85)
        ax.plot([], [], "o-", color=LAMBDA_COLORS[lam],
                label=rf"$\lambda={lam}$ (dotted: $t^*_{{\rm ref}}$)")
    ax.set_xlabel(r"threshold $\varepsilon$")
    ax.set_ylabel(r"$t_{\mathrm{first}}(\varepsilon)$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "threshold_sensitivity.png"), dpi=180)
    plt.close(fig)

    # ---- NEW: solver_convergence -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    keys = [f"N{N}_dt{_dt_tag(dt)}" for (N, dt) in REFINEMENT_CONFIGS]
    xt = [rf"${N}^2$, ${dt:g}$" for (N, dt) in REFINEMENT_CONFIGS]
    for lam in params["lambdas"]:
        ys = [ref_man["refinement"][str(lam)][k]["t_ext"] for k in keys]
        ax.plot(range(len(keys)), ys, "o-", color=LAMBDA_COLORS[lam],
                label=rf"$\lambda={lam}$")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(xt, fontsize=9)
    ax.set_xlabel(r"solver configuration (grid, $\Delta t$)")
    ax.set_ylabel(r"$t^*_{\mathrm{ref}}$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "solver_convergence.png"), dpi=180)
    plt.close(fig)

    print(f"[figures] wrote 8 figures to {fig_dir}")


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",
                        choices=["reference", "train", "analyze", "tables",
                                 "figures", "post"],
                        required=True,
                        help="'post' = analyze + tables + figures")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--lam", type=float, help="train stage only")
    parser.add_argument("--net-seed", type=int, help="train stage only")
    parser.add_argument("--sampling", default="adaptive",
                        choices=["adaptive", "uniform", "rar"])
    parser.add_argument("--print-jobs", action="store_true",
                        help="print the pre-registered training grid as "
                             "JSON and exit (used by gpu_launcher.py)")
    args = parser.parse_args()

    params = R.load_config("exp63", smoke=args.smoke)
    params["_smoke"] = args.smoke
    if args.smoke:
        print(">>> SMOKE MODE: truncated budgets, "
              "results not publication-grade.")

    if args.print_jobs:
        print(json.dumps(job_grid(params)))
        return 0
    if args.stage == "reference":
        stage_reference(params)
    elif args.stage == "train":
        if args.lam is None or args.net_seed is None:
            parser.error("--stage train requires --lam and --net-seed")
        stage_train_single(params, args.lam, args.net_seed, args.sampling)
    elif args.stage == "analyze":
        stage_analyze(params)
    elif args.stage == "tables":
        stage_tables(params)
    elif args.stage == "figures":
        stage_figures(params)
    elif args.stage == "post":
        stage_analyze(params)
        stage_tables(params)
        stage_figures(params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
