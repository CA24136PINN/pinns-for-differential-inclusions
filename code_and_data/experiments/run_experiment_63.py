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
                  val_every=250, n_val=4000, seed=0,
                  hold_steps=5000, decay_steps=3000, decay_rate=0.85,
                  adaptive_fraction=0.3, n_hard_pool=4000,
                  n_candidate_pool=20000, refine_every=500):
    """Train with the EXACT-Phi distance-residual loss (eps = 0)."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    net = build_mlp(in_dim=3, hidden=hidden, n_hidden_layers=n_hidden_layers)
    schedule = HoldThenExponentialDecay(lr, hold_steps, decay_steps,
                                        decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    # Fixed validation set (also reused by the eps-sensitivity check, F2)
    rng_val = np.random.default_rng(seed + 777)
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
        uc = u_theta(net, tc, xc, yc)
        idx = tf.argsort(tf.abs(uc[:, 0]))[:n_hard_pool]
        return (tf.gather(tc, idx), tf.gather(xc, idx), tf.gather(yc, idx))

    loss_history, val_history = [], []
    for epoch in range(1, n_epochs + 1):
        if epoch == 1 or epoch % refine_every == 0:
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


# ----------------------------------------------------------------------------
# Stage 1: TRAIN -- all heavy computation; writes results/raw/exp63/
# (raw_data.npz with every array any figure needs, manifest_63.json with
# every number any table/macro needs, checkpoints/).  NO figures here.
# ----------------------------------------------------------------------------

EXP = "exp63"


def stage_train(params):
    n_epochs = int(params["n_epochs"])
    ckpt_dir = R.checkpoints_dir(EXP)
    t_run_start = time.time()
    raw = {}

    # ---- 1. Reference solutions -----------------------------------------
    print("Reference solver sweep ...")
    ref = {}
    for lam in LAMBDA_SWEEP_REF:
        times, norms, u_fin, text, X, Y = run_reference_solver(lam)
        ref[lam] = dict(times=times, norms=norms, u_final=u_fin, text=text)
        st = f"t* = {text:.4f}" if text is not None else "not extinguished"
        print(f"  lambda = {lam}: {st}")
        raw[f"ref_times_{lam}"] = times
        raw[f"ref_norms_{lam}"] = norms
        raw[f"ref_text_{lam}"] = np.float64(text if text is not None
                                            else np.nan)
    raw["ref_ufinal_baseline"] = ref[LAMBDA_BASELINE]["u_final"]

    # ---- 2. Baseline training (lambda = 0.6, exact Phi) ------------------
    print(f"\nTraining baseline (lambda = {LAMBDA_BASELINE}, eps = 0) ...")
    net0, loss_hist0, val_hist0, val_pts0 = train_dr_pinn(
        lam=LAMBDA_BASELINE, T=T_HORIZON, n_epochs=n_epochs, seed=0)
    net0.save_weights(os.path.join(
        ckpt_dir, f"relay_lam{LAMBDA_BASELINE}.weights.h5"))  # (F3)
    val_loss0 = val_hist0[-1][1] if val_hist0 else loss_hist0[-1]
    raw["baseline_loss_hist"] = np.asarray(loss_hist0, dtype=np.float64)
    raw["baseline_val_epochs"] = np.asarray([e for e, _ in val_hist0])
    raw["baseline_val_vals"] = np.asarray([v for _, v in val_hist0])

    # Training-narrative diagnostics (auto-detected, not hand-quoted):
    plateau_window = loss_hist0[100:min(4000, len(loss_hist0))]
    plateau_level = float(np.median(plateau_window)) if plateau_window else 0.0
    escape_epoch = next(
        (i for i, v in enumerate(loss_hist0)
         if plateau_level > 0 and i > 100 and v < plateau_level / 10),
        None)

    # ---- 3. Validation vs reference (same run, same net) ------------------
    times_net, l2_net, snaps_net, X, Y = evaluate_network_on_grid(
        net0, T_HORIZON)
    r0 = ref[LAMBDA_BASELINE]
    l2_ref_interp = np.interp(times_net, r0["times"], r0["norms"])
    norm_disc = float(np.max(np.abs(l2_net - l2_ref_interp))
                      / (l2_ref_interp.max() + 1e-12))
    max_pt_err = float(np.abs(snaps_net[-1] - r0["u_final"]).max())
    post_ext_mask = times_net > (r0["text"] + 0.02) if r0["text"] else None
    post_ext_plateau = (float(np.median(l2_net[post_ext_mask]))
                        if post_ext_mask is not None and post_ext_mask.any()
                        else float("nan"))
    print(f"  norm discrepancy {norm_disc:.2%}, "
          f"max pointwise err at T: {max_pt_err:.2e}, "
          f"post-extinction plateau ~{post_ext_plateau:.1e}")
    raw["baseline_times_net"] = times_net
    raw["baseline_l2_net"] = l2_net
    raw["baseline_snap_final"] = snaps_net[-1]

    # ---- 4. Branch-selection diagnostic data (SAME net; F3) ---------------
    tf.random.set_seed(1)
    n_diag = 4000
    t_d = tf.random.uniform((n_diag, 1), 0.0, T_HORIZON)
    x_d = tf.random.uniform((n_diag, 1), 0.0, 1.0)
    y_d = tf.random.uniform((n_diag, 1), 0.0, 1.0)
    u_d, z_d = pde_operator(net0, t_d, x_d, y_d)
    s_vals = u_d.numpy().flatten()
    z_vals = z_d.numpy().flatten()
    z_front_max = float(np.max(np.abs(z_vals)))
    raw["diag_s_vals"] = s_vals
    raw["diag_z_vals"] = z_vals

    # ---- 5. eps-sensitivity check (F2) ------------------------------------
    eps_vals, eps_max_rel = eps_sensitivity(net0, LAMBDA_BASELINE, val_pts0)
    print("  eps-sensitivity:",
          {f"{e:g}": f"{v:.6e}" for e, v in eps_vals.items()},
          f"max rel dev = {eps_max_rel:.2e}")

    # ---- 6. Sweep over lambda (fixed tolerance) ----------------------------
    sweep = {}
    for lam in LAMBDA_SWEEP_MAIN:
        if lam == LAMBDA_BASELINE:
            net_l, val_l = net0, val_loss0
            times_l, l2_l = times_net, l2_net
        else:
            print(f"\nTraining lambda = {lam} ...")
            net_l, _, val_hist_l, _ = train_dr_pinn(
                lam=lam, T=T_HORIZON, n_epochs=n_epochs, seed=0,
                log_every=8000)
            net_l.save_weights(os.path.join(ckpt_dir,
                                            f"relay_lam{lam}.weights.h5"))
            val_l = val_hist_l[-1][1] if val_hist_l else float("nan")
            times_l, l2_l, _, _, _ = evaluate_network_on_grid(net_l,
                                                              T_HORIZON)
        text_net = detect_extinction(times_l, l2_l)
        sweep[lam] = dict(text_net=text_net, text_ref=ref[lam]["text"],
                          val_loss=val_l)
        raw[f"net_times_{lam}"] = times_l
        raw[f"net_l2_{lam}"] = l2_l
        print(f"  lambda={lam}: t*_ref = {ref[lam]['text']}, "
              f"t*_net = {text_net}, val loss = {val_l:.4e}")

    # ---- 7. Raw data + manifest -------------------------------------------
    R.save_raw(EXP, **raw)

    manifest = {
        "script": "run_experiment_63.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - t_run_start,
        "smoke": bool(params.get("_smoke", False)),
        "n_epochs": n_epochs,
        "eps_training": 0.0,
        "extinction_tolerance": EXTINCTION_TOL,
        "eval_time_step": T_HORIZON / (EVAL_N_TIMES - 1),
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__,
                     "scipy": scipy.__version__,
                     "tensorflow": tf.__version__},
        "hardware": R.hardware_string(),
        "reference_extinction_times": {str(l): ref[l]["text"]
                                       for l in LAMBDA_SWEEP_REF},
        "baseline": {"val_loss": float(val_loss0), "norm_disc": norm_disc,
                     "max_pt_err": max_pt_err,
                     "post_ext_plateau": post_ext_plateau,
                     "plateau_level": plateau_level,
                     "escape_epoch": escape_epoch,
                     "z_front_max": z_front_max},
        "eps_sensitivity": {"values": {str(k): v for k, v in eps_vals.items()},
                            "max_rel_dev": eps_max_rel},
        "sweep": {str(l): {"t_ref": sweep[l]["text_ref"],
                           "t_net": sweep[l]["text_net"],
                           "val_loss": float(sweep[l]["val_loss"])}
                  for l in LAMBDA_SWEEP_MAIN},
    }
    R.save_manifest(EXP, manifest, "63")
    print(f"\n[train] done in {time.time() - t_run_start:.0f} s.")


# ----------------------------------------------------------------------------
# Stage 2: TABLES -- results/aggregated/results_63.tex from the manifest.
# ----------------------------------------------------------------------------


def stage_tables():
    man = R.load_manifest(EXP, "63")
    base = man["baseline"]
    sweep = man["sweep"]

    def relerr(lam):
        tr, tn = sweep[str(lam)]["t_ref"], sweep[str(lam)]["t_net"]
        if tr is None or tn is None:
            return None
        return abs(tn - tr) / tr

    macros = {
        "RelayValLoss": R.sci_tex(base["val_loss"]),  # math-mode body
        "RelayRMS": f"{np.sqrt(base['val_loss']):.2f}",
        "RelayPlateauLevel": f"{base['plateau_level']:.0f}",
        "RelayEscapeEpoch": (f"{base['escape_epoch']:,}".replace(",", r"\,")
                             if base["escape_epoch"] else "--"),
        "RelayNormDisc": f"{base['norm_disc']*100:.2f}\\%",
        "RelayMaxPtErr": R.sci_tex(base["max_pt_err"]),
        "RelayPostExtPlateau": rf"\approx{R.sci_tex(base['post_ext_plateau'], 0)}",
        "RelayZFrontMax": f"{base['z_front_max']:.0f}",
        "RelayTRefBase": f"{man['reference_extinction_times'][str(LAMBDA_BASELINE)]:.4f}",
        "RelayEpsSens": (R.sci_tex(man["eps_sensitivity"]["max_rel_dev"])
                         if man["eps_sensitivity"]["max_rel_dev"] > 0 else "0"),
    }
    for lam, key in zip(LAMBDA_SWEEP_MAIN, ["A", "B", "C"]):
        s = sweep[str(lam)]
        macros[f"RelayTRef{key}"] = f"{s['t_ref']:.4f}"
        macros[f"RelayTNet{key}"] = (f"{s['t_net']:.4f}"
                                     if s["t_net"] else "--")
        re_ = relerr(lam)
        macros[f"RelayRelErr{key}"] = (f"${re_*100:.1f}\\%$"
                                       if re_ is not None else "--")
    abs_errs = [abs(sweep[str(l)]["t_net"] - sweep[str(l)]["t_ref"])
                for l in LAMBDA_SWEEP_MAIN
                if sweep[str(l)]["t_net"] and sweep[str(l)]["t_ref"]]
    if abs_errs:
        macros["RelayAbsErrMin"] = f"{min(abs_errs):.3f}"
        macros["RelayAbsErrMax"] = f"{max(abs_errs):.3f}"

    out = R.aggregated_dir()
    R.write_macros(os.path.join(out, "results_63.tex"), macros,
                   "run_experiment_63.py --stage tables")
    with open(os.path.join(out, "manifest_63.json"), "w") as f:
        json.dump(man, f, indent=2)


# ----------------------------------------------------------------------------
# Stage 3: FIGURES -- all Section 6.3 figures from raw_data.npz (seconds,
# no TensorFlow computation, no retraining).
# ----------------------------------------------------------------------------


def _ref(raw, lam):
    text = float(raw[f"ref_text_{lam}"])
    return (raw[f"ref_times_{lam}"], raw[f"ref_norms_{lam}"],
            None if np.isnan(text) else text)


def plot_phi_graph(ax, lam, s_min, s_max, lw=2, label=r"graph of $\Phi$"):
    """Three-branch relay graph on [s_min, s_max]."""
    if s_max > 0:
        ax.plot([max(s_min, 0), s_max], [-lam, -lam], color="tab:red", lw=lw,
                label=label, zorder=3)
        label = None
    if s_min < 0:
        ax.plot([s_min, min(s_max, 0)], [lam, lam], color="tab:red", lw=lw,
                label=label, zorder=3)
        label = None
    ax.plot([0, 0], [-lam, lam], color="tab:red", lw=lw, label=label, zorder=3)


def stage_figures():
    raw = R.load_raw(EXP)
    fig_dir = R.figures_dir()

    # ---- Figure: reference_extinction_curves ------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for lam in LAMBDA_SWEEP_REF:
        times, norms, text = _ref(raw, lam)
        label = rf"$\lambda={lam}$" + (" (heat eq.)" if lam == 0.0 else "")
        ax.semilogy(times, np.maximum(norms, 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2, label=label)
        if text is not None:
            ax.axvline(text, color=LAMBDA_COLORS[lam], ls=":", lw=0.9,
                       alpha=0.8)
            ax.annotate(rf"$t^*={text:.4f}$",
                        xy=(text, NORM_FLOOR * 4),
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

    # ---- Figure: training_history_relay -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(raw["baseline_loss_hist"], color="lightsteelblue", lw=0.8,
                label="training loss (per step)", rasterized=True)
    ax.semilogy(raw["baseline_val_epochs"], raw["baseline_val_vals"],
                color="tab:red", lw=1.6, marker="o", markersize=2.5,
                label="validation loss (fixed batch)")
    ax.axvline(5000, color="0.4", ls="--", lw=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{incl}}(\theta)$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "training_history_relay.png"), dpi=180)
    plt.close(fig)

    # ---- Figure: dr_pinn_vs_reference -------------------------------------
    times_ref, norms_ref, text_ref = _ref(raw, LAMBDA_BASELINE)
    times_net = raw["baseline_times_net"]
    l2_net = raw["baseline_l2_net"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(times_ref, np.maximum(norms_ref, 1e-16),
                     color="tab:blue", label="Reference solution", lw=2)
    axes[0].semilogy(times_net, np.maximum(l2_net, 1e-16), "--",
                     color="tab:orange", label="DR-PINN", lw=2)
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
    axes[0].legend(loc="lower left")
    im = axes[1].imshow(
        np.abs(raw["baseline_snap_final"] - raw["ref_ufinal_baseline"]).T,
        origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    axes[1].set_title(rf"$|u_\theta - u_{{\mathrm{{ref}}}}|$ at $t=T={T_HORIZON}$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dr_pinn_vs_reference.png"), dpi=180)
    plt.close(fig)

    # ---- Figure: branch_selection_diagnostic (with zoom inset) ------------
    # Restored to the notebook layout: main cloud over the graph of Phi,
    # plus an inset magnifying the neighbourhood of the extinction front
    # (s near 0), where the paper text discusses the vertical scatter
    # caused by the jump of Phi.
    s_vals = raw["diag_s_vals"]
    z_vals = raw["diag_z_vals"]
    lam = LAMBDA_BASELINE
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(s_vals, z_vals, s=5, alpha=0.20, color="tab:blue",
               edgecolors="none", label="Collocation points", rasterized=True)
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

    # ---- Figure: extinction_time_sweep -------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in LAMBDA_SWEEP_MAIN:
        times_r, norms_r, text_r = _ref(raw, lam)
        ax.semilogy(times_r, np.maximum(norms_r, 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2,
                    label=rf"$\lambda={lam}$ (reference)")
        ax.semilogy(raw[f"net_times_{lam}"],
                    np.maximum(raw[f"net_l2_{lam}"], 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2, ls="--",
                    label=rf"$\lambda={lam}$ (DR-PINN)")
        if text_r is not None:
            ax.axvline(text_r, color=LAMBDA_COLORS[lam], ls=":",
                       lw=0.9, alpha=0.7)
    ax.set_ylim(NORM_FLOOR, 2.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=10, ncol=2, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "extinction_time_sweep.png"), dpi=180)
    plt.close(fig)

    print(f"[figures] wrote 5 figures to {fig_dir}")


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "tables", "figures",
                                            "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.stage in ("train", "all"):
        params = R.load_config("exp63", smoke=args.smoke)
        params["_smoke"] = args.smoke
        if args.smoke:
            print(">>> SMOKE MODE: truncated budgets, "
                  "results not publication-grade.")
        stage_train(params)
    if args.stage in ("tables", "all"):
        stage_tables()
    if args.stage in ("figures", "all"):
        stage_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
