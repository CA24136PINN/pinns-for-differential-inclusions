"""
Section 6.1 replication script: linear control system with polytopic input set.

Single-run, clean-state pipeline. One execution of this script produces, in a
single documented run:

  figures/  control_set.png, loss_single.png, trajectory_vs_tube.png,
            scalability.png
  generated/results_61.tex   -- every number quoted in Section 6.1 of the
                                manuscript, as LaTeX macros
  generated/manifest_61.json -- seeds, library versions, hardware, timings,
                                and all reported values

Fixes relative to the previous notebook (referee report, Example 6.1):

  (F1) Reference-tube integrator: each piecewise-constant control segment is
       integrated over its exact interval [t_j, t_{j+1}] with dense output;
       the state handed to the next segment is the state at *exactly* the
       switching time t_{j+1}.  (The old code carried the state at the last
       global-grid point inside the segment, introducing O(5e-2) errors.)
  (F2) Built-in integrator self-check: the piecewise RK45 propagation is
       compared against the closed-form matrix-exponential propagation
       x(tb) = e^{A dt} x(ta) + A^{-1}(e^{A dt}-I) B u; the run aborts if the
       maximum deviation exceeds 1e-7.
  (F3) Reference grid unified at n_ref_t = 200 points (paper and code agree).
  (F4) Independent dense-grid inclusion-residual validation (mean and max
       distance dist(xdot - A x, B U~) at N_dense points off the collocation
       grid, for every ensemble member), analogous to Section 6.2.
  (F5) Scalability timings: one discarded warm-up run, then N_REPS repetitions
       per configuration; mean +/- std reported and plotted as error bars.
  (F6) All quantitative statements exported as LaTeX macros so that text,
       tables and figures provably come from this one run.

Usage:  python run_experiment_61.py            # full run
        python run_experiment_61.py --smoke    # tiny smoke test (~1 min)
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.spatial
from scipy.linalg import expm
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import repro_utils as R  # noqa: E402
from paper_style import apply_paper_style  # noqa: E402

apply_paper_style()

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


@dataclass
class Config:
    # system
    T: float = 1.5
    d: int = 2
    # training
    seed: int = 42
    n_col: int = 300
    width: int = 64
    depth: int = 3
    lr: float = 1e-3
    epochs: int = 1000
    # ensemble
    n_ensemble: int = 5
    # reference tube  (F3: unified with the paper at 200 points)
    n_ref_random: int = 300
    n_ref_t: int = 200
    # dense-grid residual validation (F4)
    n_dense: int = 5000
    # scalability (F5)
    timing_epochs: int = 50
    timing_reps: int = 5


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------------
# Problem data (identical to the manuscript)
# ----------------------------------------------------------------------------

A_np = np.array([[-0.5, 1.0], [-1.0, -0.5]], dtype=np.float64)
B_np = np.eye(2, dtype=np.float64)
x0 = np.array([1.0, 0.0], dtype=np.float64)

U_extreme = np.array(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float64
)
rng_init = np.random.default_rng(0)
U_interior = rng_init.uniform(-0.35, 0.35, size=(8, 2))
U_raw = np.vstack([U_extreme, U_interior])  # K_raw = 12
K_raw = len(U_raw)


# ----------------------------------------------------------------------------
# Quickhull preprocessing
# ----------------------------------------------------------------------------


def compute_extreme_vertices(points):
    K, d = points.shape
    if K <= 1:
        return points.copy()
    if d == 1:
        return points[[np.argmin(points[:, 0]), np.argmax(points[:, 0])]]
    try:
        hull = scipy.spatial.ConvexHull(points)
        return points[hull.vertices]
    except scipy.spatial.qhull.QhullError:
        return points.copy()


# ----------------------------------------------------------------------------
# QP projection
# ----------------------------------------------------------------------------


def qp_project(v, vertices):
    K = len(vertices)
    if K == 1:
        return vertices[0].copy()
    G = vertices @ vertices.T
    c = vertices @ v
    res = scipy.optimize.minimize(
        fun=lambda lam: float(lam @ G @ lam) - 2.0 * float(c @ lam),
        x0=np.ones(K) / K,
        jac=lambda lam: 2.0 * (G @ lam) - 2.0 * c,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * K,
        constraints={"type": "eq", "fun": lambda lam: lam.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 300},
    )
    lam = np.clip(res.x, 0.0, 1.0)
    lam /= lam.sum()
    return vertices.T @ lam


def project_batch(v_batch, vertices):
    projs = np.empty_like(v_batch)
    for i in range(v_batch.shape[0]):
        projs[i] = qp_project(v_batch[i], vertices)
    return projs


# ----------------------------------------------------------------------------
# Network and loss (unchanged relative to the previous, already-correct
# implementation: the state-dependent translation A x stays in the graph)
# ----------------------------------------------------------------------------


def build_network(d_out, width, depth):
    inp = tf.keras.Input(shape=(1,))
    z = inp
    for _ in range(depth):
        z = tf.keras.layers.Dense(
            width,
            activation="tanh",
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
        )(z)
    out = tf.keras.layers.Dense(
        d_out,
        activation=None,
        kernel_initializer="glorot_normal",
        bias_initializer="zeros",
    )(z)
    return tf.keras.Model(inp, out)


class TrialSolution(tf.keras.Model):
    def __init__(self, x0_, width, depth):
        super().__init__()
        self.x0 = tf.constant(x0_, dtype=tf.float32)
        self.d = len(x0_)
        self.N = build_network(d_out=self.d, width=width, depth=depth)

    def call(self, t, training=False):
        return self.x0 + t * self.N(t, training=training)


def state_and_velocity(model, t_col):
    with tf.GradientTape() as tape:
        tape.watch(t_col)
        x = model(t_col, training=True)
    dx = tape.batch_jacobian(x, t_col)[:, :, 0]
    return x, dx


def compute_dr_loss(model, t_col, A, BU_ext):
    with tf.GradientTape() as tape:
        tape.watch(t_col)
        x = model(t_col, training=True)
    dx = tape.batch_jacobian(x, t_col)[:, :, 0]
    A_tf = tf.constant(A, dtype=dx.dtype)
    Ax = tf.linalg.matmul(x, A_tf, transpose_b=True)
    q = dx - Ax
    q_np = q.numpy().astype(np.float64)
    control_projs = project_batch(q_np, BU_ext)
    control_projs_tf = tf.constant(control_projs, dtype=dx.dtype)
    diff = q - control_projs_tf
    return tf.reduce_mean(tf.reduce_sum(diff**2, axis=-1))


def train_dr_pinn(model, cfg, A, BU_ext, verbose=True):
    t_col = tf.cast(
        tf.reshape(tf.linspace(0.0, cfg.T, cfg.n_col), (-1, 1)), tf.float32
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    history = {"loss": [], "time_per_epoch": []}
    log_every = max(1, cfg.epochs // 10)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        with tf.GradientTape() as tape:
            loss = compute_dr_loss(model, t_col, A, BU_ext)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        history["loss"].append(float(loss))
        history["time_per_epoch"].append(time.perf_counter() - t0)
        if verbose and (epoch == 1 or epoch % log_every == 0):
            print(f"  Epoch {epoch:5d} | loss = {float(loss):.4e}")
    return history


# ----------------------------------------------------------------------------
# Reference reachable tube  --  CORRECTED integrator (F1) + self-check (F2)
# ----------------------------------------------------------------------------


def _exact_segment(A, B, x, u, dt):
    """Closed-form solution of xdot = A x + B u on [0, dt] (u constant),
    valid for invertible A:  x(dt) = e^{A dt} x + A^{-1}(e^{A dt}-I) B u."""
    E = expm(A * dt)
    return E @ x + np.linalg.solve(A, (E - np.eye(A.shape[0])) @ (B @ u))


def _integrate_piecewise(A, B, x_start, ts, controls, t_grid, method="rk45"):
    """Integrate a piecewise-constant-control trajectory on the global grid.

    Each segment [ts[j], ts[j+1]] is integrated over its EXACT interval; the
    state handed to the next segment is the state at exactly ts[j+1]
    (fix F1).  method='exact' uses the closed-form matrix-exponential
    propagation and serves as the self-check oracle (F2).
    """
    x_curr = x_start.copy()
    traj = np.zeros((len(t_grid), len(x_start)))
    for j in range(len(controls)):
        ta, tb = ts[j], ts[j + 1]
        u = controls[j]
        mask = (t_grid >= ta - 1e-12) & (t_grid <= tb + 1e-12)
        if method == "rk45":
            sol = scipy.integrate.solve_ivp(
                lambda t, x, u=u: A @ x + B @ u,
                [ta, tb],
                x_curr,
                dense_output=True,
                method="RK45",
                rtol=1e-9,
                atol=1e-11,
            )
            if mask.any():
                traj[mask] = sol.sol(t_grid[mask]).T
            x_curr = sol.y[:, -1].copy()  # solve_ivp ends exactly at tb
        else:  # exact matrix-exponential propagation
            for idx in np.where(mask)[0]:
                traj[idx] = _exact_segment(A, B, x_curr, u, t_grid[idx] - ta)
            x_curr = _exact_segment(A, B, x_curr, u, tb - ta)
    return traj


def compute_reference_tube(A, B, U_vertices, x0_, T, n_random, n_t, seed=0):
    """Sampled reference cloud: 4 constant bang-bang + n_random random
    piecewise-constant trajectories on a uniform n_t-point grid.

    Returns (t_grid, tube, selfcheck) where selfcheck is the maximum
    deviation between the RK45 and the exact matrix-exponential propagation
    over all trajectories and grid times (F2)."""
    rng_ref = np.random.default_rng(seed)
    K = len(U_vertices)
    t_grid = np.linspace(0.0, T, n_t)
    trajectories = []
    selfcheck = 0.0

    # Family 1: constant bang-bang
    for k in range(K):
        ts = np.array([0.0, T])
        controls = [U_vertices[k]]
        tr = _integrate_piecewise(A, B, x0_, ts, controls, t_grid, "rk45")
        te = _integrate_piecewise(A, B, x0_, ts, controls, t_grid, "exact")
        selfcheck = max(selfcheck, float(np.abs(tr - te).max()))
        trajectories.append(tr)

    # Family 2: piecewise-constant random controls
    for _ in range(n_random):
        n_seg = rng_ref.integers(3, 12)
        ts = np.concatenate(
            [[0.0], np.sort(rng_ref.uniform(0.0, T, n_seg - 1)), [T]]
        )
        controls = [U_vertices[k] for k in rng_ref.integers(0, K, size=n_seg)]
        tr = _integrate_piecewise(A, B, x0_, ts, controls, t_grid, "rk45")
        te = _integrate_piecewise(A, B, x0_, ts, controls, t_grid, "exact")
        selfcheck = max(selfcheck, float(np.abs(tr - te).max()))
        trajectories.append(tr)

    tube = np.array(trajectories)  # (4 + n_random, n_t, d)
    assert selfcheck < 1e-7, (
        f"Integrator self-check FAILED: max |RK45 - exact| = {selfcheck:.3e}"
    )
    return t_grid, tube, selfcheck


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def one_sided_hausdorff(A_pts, B_pts):
    diffs = A_pts[:, np.newaxis, :] - B_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    return float(np.max(np.min(dists, axis=1)))


def dense_grid_residual(model, cfg, A, BU_ext, seed=12345):
    """(F4) Independent inclusion-residual validation: mean and max
    dist(xdot - A x, B U~) at n_dense uniform-random times OFF the
    collocation grid."""
    rng = np.random.default_rng(seed)
    t_dense = np.sort(rng.uniform(0.0, cfg.T, cfg.n_dense))
    t_tf = tf.cast(t_dense.reshape(-1, 1), tf.float32)
    x, dx = state_and_velocity(model, t_tf)
    q = (dx - tf.linalg.matmul(x, tf.constant(A, dtype=dx.dtype), transpose_b=True))
    q_np = q.numpy().astype(np.float64)
    projs = project_batch(q_np, BU_ext)
    d = np.linalg.norm(q_np - projs, axis=1)
    return float(d.mean()), float(d.max())



# ----------------------------------------------------------------------------
# Scalability study (F5): warm-up + repetitions, mean +/- std
# ----------------------------------------------------------------------------


def timed_training_ms(cfg_sc, A, BU, reps, x0_):
    """One discarded warm-up run, then `reps` timed runs; returns
    (mean_ms_per_epoch, std_ms_per_epoch)."""
    means = []
    for r in range(reps + 1):  # r = 0 is the warm-up, discarded
        set_seed(1000 + r)
        m = TrialSolution(x0_, width=cfg_sc.width, depth=cfg_sc.depth)
        h = train_dr_pinn(m, cfg_sc, A, BU, verbose=False)
        if r > 0:
            means.append(np.mean(h["time_per_epoch"]) * 1e3)
    return float(np.mean(means)), float(np.std(means))


def scalability_study(cfg):
    results = {"K": {}, "d": {}}

    # --- sweep in K_raw at fixed d = 2 -----------------------------------
    K_values = [8, 16, 32, 64, 128]
    for K_r in K_values:
        rng_sc = np.random.default_rng(K_r)
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        U_bnd = np.column_stack([np.cos(angles), np.sin(angles)])
        U_int = rng_sc.uniform(-0.4, 0.4, (max(K_r - 6, 0), 2))
        U_sc = np.vstack([U_bnd, U_int])[:K_r]
        BU_sc_raw = (B_np @ U_sc.T).T
        BU_sc_ext = compute_extreme_vertices(BU_sc_raw)

        cfg_sc = Config(epochs=cfg.timing_epochs, n_col=100,
                        width=cfg.width, depth=cfg.depth)
        with_m, with_s = timed_training_ms(cfg_sc, A_np, BU_sc_ext,
                                           cfg.timing_reps, x0)
        wo_m, wo_s = timed_training_ms(cfg_sc, A_np, BU_sc_raw,
                                       cfg.timing_reps, x0)
        results["K"][K_r] = dict(K_tilde=len(BU_sc_ext),
                                 with_qh=(with_m, with_s),
                                 without_qh=(wo_m, wo_s))
        print(f"  K_raw={K_r:3d} K~={len(BU_sc_ext):2d} "
              f"| w/QH {with_m:7.1f}+-{with_s:5.1f} ms "
              f"| w/o {wo_m:7.1f}+-{wo_s:5.1f} ms")

    # --- sweep in d at fixed K_raw = 32 ----------------------------------
    d_values = [2, 4, 6, 8]
    for d in d_values:
        rng_d = np.random.default_rng(d + 100)
        A_d = rng_d.standard_normal((d, d)) * 0.3
        A_d -= (np.abs(A_d).sum(axis=1)[:, None] + 0.5) * np.eye(d)
        B_d = np.eye(d)
        x0_d = np.zeros(d)
        x0_d[0] = 1.0
        U_d = rng_d.uniform(-1.0, 1.0, (32, d))
        BU_d_raw = (B_d @ U_d.T).T
        BU_d_ext = compute_extreme_vertices(BU_d_raw)

        cfg_d = Config(d=d, epochs=cfg.timing_epochs, n_col=100,
                       width=cfg.width, depth=cfg.depth)
        with_m, with_s = timed_training_ms(cfg_d, A_d, BU_d_ext,
                                           cfg.timing_reps, x0_d)
        wo_m, wo_s = timed_training_ms(cfg_d, A_d, BU_d_raw,
                                       cfg.timing_reps, x0_d)
        results["d"][d] = dict(K_tilde=len(BU_d_ext),
                               with_qh=(with_m, with_s),
                               without_qh=(wo_m, wo_s))
        print(f"  d={d} K~={len(BU_d_ext):2d} "
              f"| w/QH {with_m:7.1f}+-{with_s:5.1f} ms "
              f"| w/o {wo_m:7.1f}+-{wo_s:5.1f} ms")
    return results


# ----------------------------------------------------------------------------
# Finite-difference gradient check for the dist^2(xdot - Ax, BU) loss
# (referee item: verify the custom projection backward pass in 6.1).
# Uses a small freshly initialised model; no retraining needed.
# ----------------------------------------------------------------------------


def gradient_check_61(cfg, A, BU_ext, n_params=15, fd_eps=1e-4, seed=1234):
    set_seed(seed)
    model = TrialSolution(x0, width=16, depth=2)
    rng = np.random.default_rng(seed)
    t_c = tf.constant(rng.uniform(0.0, cfg.T, (64, 1)), tf.float32)

    def loss_fn():
        with tf.GradientTape() as tape:
            tape.watch(t_c)
            xt = model(t_c, training=False)
        dx = tape.batch_jacobian(xt, t_c)[:, :, 0]
        q = dx - tf.linalg.matmul(
            xt, tf.constant(A, dtype=dx.dtype), transpose_b=True)
        proj = tf.stop_gradient(tf.constant(
            project_batch(q.numpy().astype(np.float64), BU_ext),
            dtype=q.dtype))
        return tf.reduce_mean(tf.reduce_sum((q - proj) ** 2, axis=1))

    # NOTE on the projection: dist^2 to a convex set is C^1 with gradient
    # 2 (q - P(q)); differentiating through a FROZEN projection point is
    # exact almost everywhere, so autodiff and finite differences must
    # agree away from face switches.  We recompute the projection at each
    # perturbed point in the FD pass (true loss), and compare against the
    # frozen-projection autodiff gradient.
    def true_loss():
        with tf.GradientTape() as tape:
            tape.watch(t_c)
            xt = model(t_c, training=False)
        dx = tape.batch_jacobian(xt, t_c)[:, :, 0]
        q = (dx - xt.numpy() @ A.T).numpy().astype(np.float64)
        d = q - project_batch(q, BU_ext)
        return float(np.mean(np.sum(d ** 2, axis=1)))

    with tf.GradientTape() as tape:
        val = loss_fn()
    grads = tape.gradient(val, model.trainable_variables)

    flat = []
    for v, g in zip(model.trainable_variables, grads):
        gn = np.zeros_like(v.numpy()) if g is None else g.numpy()
        for idx in np.ndindex(tuple(v.shape)):
            flat.append((v, idx, float(gn[idx])))
    # Restrict to parameters with non-negligible gradient: central FD in a
    # float32 forward pass cannot resolve directions with |g| near zero
    # (cancellation noise), which would only test numerical noise, not the
    # backward pass.  The threshold is recorded in the manifest.
    g_floor = 1e-3 * max(abs(g) for _, _, g in flat)
    flat = [f for f in flat if abs(f[2]) >= g_floor]
    rng.shuffle(flat)
    checked, rel_errs = [], []
    for v, idx, g_ad in flat[:n_params]:
        orig = float(v.numpy()[idx])
        for sgn, store in ((+1, "p"), (-1, "m")):
            arr = v.numpy()
            arr[idx] = orig + sgn * fd_eps
            v.assign(arr)
            if sgn > 0:
                fp = true_loss()
            else:
                fm = true_loss()
        arr = v.numpy(); arr[idx] = orig; v.assign(arr)
        g_fd = (fp - fm) / (2 * fd_eps)
        denom = max(abs(g_ad), abs(g_fd), 1e-10)
        rel_errs.append(abs(g_ad - g_fd) / denom)
        checked.append({"param": getattr(v, "path", None) or v.name,
                        "index": list(idx),
                        "autodiff": g_ad, "fd": g_fd,
                        "rel_err": rel_errs[-1]})
    return {"n_params": len(checked), "fd_eps": fd_eps,
            "grad_floor_relative": 1e-3,
            "max_rel_err": float(np.max(rel_errs)),
            "median_rel_err": float(np.median(rel_errs)),
            "details": checked}


# ----------------------------------------------------------------------------
# Stage 1: TRAIN -- all heavy computation; writes results/raw/exp61/
# (raw_data.npz + manifest_61.json).  NO figures here.
# ----------------------------------------------------------------------------

EXP = "exp61"


def stage_train(cfg, smoke=False):
    t_run_start = time.time()
    raw = {}

    # ---- Quickhull preprocessing -----------------------------------------
    BU_raw = (B_np @ U_raw.T).T
    BU_extreme = compute_extreme_vertices(BU_raw)
    K_tilde = len(BU_extreme)
    print(f"Quickhull: K_raw = {K_raw} -> K_tilde = {K_tilde}")
    raw["U_raw"] = U_raw
    raw["BU_raw"] = BU_raw
    raw["BU_extreme"] = BU_extreme

    # ---- Single DR-PINN training ------------------------------------------
    print("Training single DR-PINN ...")
    set_seed(cfg.seed)
    model_single = TrialSolution(x0, width=cfg.width, depth=cfg.depth)
    hist = train_dr_pinn(model_single, cfg, A_np, BU_extreme)
    loss_init, loss_final = hist["loss"][0], hist["loss"][-1]
    raw["loss_hist"] = np.asarray(hist["loss"], dtype=np.float64)

    # ---- Reference tube (corrected, self-checked) -------------------------
    print("Computing reference reachable tube (corrected integrator) ...")
    t_ref, tube, selfcheck = compute_reference_tube(
        A_np, B_np, U_extreme, x0, cfg.T,
        n_random=cfg.n_ref_random, n_t=cfg.n_ref_t, seed=0,
    )
    print(f"  {tube.shape[0]} trajectories; "
          f"integrator self-check max|RK45-exact| = {selfcheck:.2e}")
    raw["t_ref"] = t_ref
    raw["tube"] = tube

    # Evaluate the single model on the same reference grid
    t_eval = tf.cast(t_ref.reshape(-1, 1), tf.float32)
    raw["x_pred_single"] = model_single(t_eval, training=False).numpy()

    # ---- Ensemble, Hausdorff ----------------------------------------------
    print(f"Training ensemble of {cfg.n_ensemble} DR-PINNs ...")
    ensemble_trajs, resid_means, resid_maxes = [], [], []
    seed_list, member_final_losses = [], []
    for s in range(cfg.n_ensemble):
        seed_s = cfg.seed + s * 100
        seed_list.append(seed_s)
        set_seed(seed_s)
        m = TrialSolution(x0, width=cfg.width, depth=cfg.depth)
        print(f"  [seed {seed_s}]")
        hist_s = train_dr_pinn(m, cfg, A_np, BU_extreme, verbose=False)
        member_final_losses.append(float(hist_s["loss"][-1]))
        ensemble_trajs.append(m(t_eval, training=False).numpy())
        # (F4) dense-grid residual per member
        rm, rx = dense_grid_residual(m, cfg, A_np, BU_extreme,
                                     seed=54321 + s)
        resid_means.append(rm)
        resid_maxes.append(rx)
        print(f"    final loss {member_final_losses[-1]:.2e}, "
              f"dense-grid residual: mean {rm:.2e}, max {rx:.2e}")
    ensemble_arr = np.array(ensemble_trajs)  # (n_ens, n_ref_t, 2)
    raw["ensemble_arr"] = ensemble_arr

    hd = np.zeros(len(t_ref))
    for ti in range(len(t_ref)):
        hd[ti] = one_sided_hausdorff(ensemble_arr[:, ti, :], tube[:, ti, :])
    mean_dh = float(hd.mean())
    print(f"Mean one-sided Hausdorff d_H+ (pooled) = {mean_dh:.4f}")

    # Per-seed distance-to-tube (mean/max over time of the distance from
    # the member's trajectory to the reference tube point cloud) -- basis
    # of the per-seed table requested by the referee.
    member_dist_mean, member_dist_max = [], []
    for e in range(ensemble_arr.shape[0]):
        d_t = np.array([
            np.min(np.linalg.norm(tube[:, ti, :]
                                  - ensemble_arr[e, ti, :], axis=1))
            for ti in range(len(t_ref))])
        member_dist_mean.append(float(d_t.mean()))
        member_dist_max.append(float(d_t.max()))

    def med_range(v):
        return {"median": float(np.median(v)), "min": float(np.min(v)),
                "max": float(np.max(v))}

    resid_mean_worst = float(np.max(resid_means))
    resid_max_worst = float(np.max(resid_maxes))

    # (referee) finite-difference check of the projection backward pass
    print("Finite-difference gradient check of dist^2 loss ...")
    gc = gradient_check_61(cfg, A_np, BU_extreme)
    print(f"  {gc['n_params']} random parameters: "
          f"max rel err = {gc['max_rel_err']:.2e}, "
          f"median = {gc['median_rel_err']:.2e}")

    # ---- Scalability (F5) --------------------------------------------------
    print("Scalability study (warm-up + repetitions) ...")
    sc = scalability_study(cfg)
    K_values = sorted(sc["K"].keys())
    d_values = sorted(sc["d"].keys())
    raw["sc_K_values"] = np.asarray(K_values)
    raw["sc_K_with_mean"] = np.asarray([sc["K"][k]["with_qh"][0]
                                        for k in K_values])
    raw["sc_K_with_std"] = np.asarray([sc["K"][k]["with_qh"][1]
                                       for k in K_values])
    raw["sc_K_without_mean"] = np.asarray([sc["K"][k]["without_qh"][0]
                                           for k in K_values])
    raw["sc_K_without_std"] = np.asarray([sc["K"][k]["without_qh"][1]
                                          for k in K_values])
    raw["sc_d_values"] = np.asarray(d_values)
    raw["sc_d_with_mean"] = np.asarray([sc["d"][d]["with_qh"][0]
                                        for d in d_values])
    raw["sc_d_with_std"] = np.asarray([sc["d"][d]["with_qh"][1]
                                       for d in d_values])
    raw["sc_d_without_mean"] = np.asarray([sc["d"][d]["without_qh"][0]
                                           for d in d_values])
    raw["sc_d_without_std"] = np.asarray([sc["d"][d]["without_qh"][1]
                                          for d in d_values])

    R.save_raw(EXP, **raw)

    manifest = {
        "script": "run_experiment_61.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - t_run_start,
        "smoke": bool(smoke),
        "config": asdict(cfg),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tensorflow": tf.__version__,
        },
        "hardware": R.hardware_string(),
        "quickhull": {"K_raw": int(K_raw), "K_tilde": int(K_tilde)},
        "training": {"loss_init": float(loss_init),
                     "loss_final": float(loss_final)},
        "reference_tube": {
            "n_trajectories": int(tube.shape[0]),
            "n_ref_t": cfg.n_ref_t,
            "integrator_selfcheck_max_abs": float(selfcheck),
        },
        "hausdorff": {"mean_dh_plus": mean_dh},
        "per_seed": {
            "seeds": seed_list,
            "final_loss": member_final_losses,
            "final_loss_agg": med_range(member_final_losses),
            "dense_resid_mean": [float(v) for v in resid_means],
            "dense_resid_mean_agg": med_range(resid_means),
            "dense_resid_max": [float(v) for v in resid_maxes],
            "dense_resid_max_agg": med_range(resid_maxes),
            "dist_to_tube_mean": member_dist_mean,
            "dist_to_tube_mean_agg": med_range(member_dist_mean),
            "dist_to_tube_max": member_dist_max,
        },
        "gradient_check": {k: v for k, v in gc.items() if k != "details"},
        "gradient_check_details": gc["details"],
        "dense_residual": {
            "n_dense": cfg.n_dense,
            "per_member_mean": [float(v) for v in resid_means],
            "per_member_max": [float(v) for v in resid_maxes],
            "worst_mean": resid_mean_worst,
            "worst_max": resid_max_worst,
        },
        "scalability": sc,
    }
    R.save_manifest(EXP, manifest, "61")
    print(f"\n[train] done in {time.time() - t_run_start:.0f} s.")


# ----------------------------------------------------------------------------
# Stage 2: TABLES -- results/aggregated/results_61.tex from the manifest.
# ----------------------------------------------------------------------------


def stage_tables():
    man = R.load_manifest(EXP, "61")
    sc = man["scalability"]
    cfg = man["config"]
    K_values = sorted(sc["K"].keys(), key=int)
    K_last = K_values[-1]
    slowdown = (sc["K"][K_last]["without_qh"][0]
                / sc["K"][K_last]["with_qh"][0])
    qh_means = [sc["K"][k]["with_qh"][0] for k in K_values]

    def ms(x):
        return f"{x:,.0f}".replace(",", r"\,")

    macros = {
        "LCLossInit": R.sci_tex(man["training"]["loss_init"]),
        "LCLossFinal": R.sci_tex(man["training"]["loss_final"]),
        "LCIntSelfCheck": R.sci_tex(
            man["reference_tube"]["integrator_selfcheck_max_abs"]),
        "LCMeanDH": f"{man['hausdorff']['mean_dh_plus']:.4f}",
        "LCnDense": f"{cfg['n_dense']:,}".replace(",", r"\,"),
        "LCResidMeanDense": R.sci_tex(man["dense_residual"]["worst_mean"]),
        "LCResidMaxDense": R.sci_tex(man["dense_residual"]["worst_max"]),
        "LCtimingReps": str(cfg["timing_reps"]),
        "LCtimeQHmin": f"${ms(min(qh_means))}$\\,ms",
        "LCtimeQHmax": f"${ms(max(qh_means))}$\\,ms",
        "LCtimeNoQHfirst":
            f"${ms(sc['K'][K_values[0]]['without_qh'][0])}$\\,ms",
        "LCtimeNoQHlast":
            f"${sc['K'][K_last]['without_qh'][0]/1000:.2f}$\\,s",
        "LCslowdown": rf"{slowdown:.1f}\times",
        "LCtimeQHdSix": f"${sc['d']['6']['with_qh'][0]/1000:.2f}$\\,s"
        if "6" in sc["d"] else "--",
        "LCtimeNoQHdSix": f"${sc['d']['6']['without_qh'][0]/1000:.2f}$\\,s"
        if "6" in sc["d"] else "--",
        "LCtimeQHdEight": f"${sc['d']['8']['with_qh'][0]/1000:.2f}$\\,s"
        if "8" in sc["d"] else "--",
        "LCtimeNoQHdEight": f"${sc['d']['8']['without_qh'][0]/1000:.2f}$\\,s"
        if "8" in sc["d"] else "--",
        "LCHardware": man["hardware"].replace("_", r"\_"),
    }
    ps = man.get("per_seed")
    if ps:
        macros["LCSeedList"] = ", ".join(str(x) for x in ps["seeds"])
        la = ps["final_loss_agg"]
        macros["LCLossMedian"] = R.sci_tex(la["median"])
        macros["LCLossRange"] = (rf"[{R.sci_tex(la['min'])},\,"
                                 rf"{R.sci_tex(la['max'])}]")
        ra = ps["dense_resid_mean_agg"]
        macros["LCResidMeanMedian"] = R.sci_tex(ra["median"])
        macros["LCResidMeanRange"] = (rf"[{R.sci_tex(ra['min'])},\,"
                                      rf"{R.sci_tex(ra['max'])}]")
        da = ps["dist_to_tube_mean_agg"]
        macros["LCDistTubeMedian"] = f"{da['median']:.4f}"
        macros["LCDistTubeRange"] = f"[{da['min']:.4f}, {da['max']:.4f}]"
    gc = man.get("gradient_check")
    if gc:
        macros["LCGradCheckMaxRel"] = R.sci_tex(gc["max_rel_err"])
        macros["LCGradCheckNParams"] = str(gc["n_params"])
        macros["LCGradCheckEps"] = R.sci_tex(gc["fd_eps"], 0)
    out = R.aggregated_dir()
    R.write_macros(os.path.join(out, "results_61.tex"), macros,
                   "run_experiment_61.py --stage tables")
    # per-seed tabular (referee: report each of the seeds explicitly)
    if ps:
        path = os.path.join(out, "table_61_seeds.tex")
        with open(path, "w") as f:
            f.write("% AUTO-GENERATED -- per-seed results, Example 6.1\n")
            f.write("\\begin{tabular}{ccccc}\n\\toprule\n")
            f.write("seed & final loss & dense resid.\\ mean & "
                    "dense resid.\\ max & mean dist.\\ to tube "
                    "\\\\\n\\midrule\n")
            for i, sd in enumerate(ps["seeds"]):
                f.write(
                    f"{sd} & ${R.sci_tex(ps['final_loss'][i])}$ & "
                    f"${R.sci_tex(ps['dense_resid_mean'][i])}$ & "
                    f"${R.sci_tex(ps['dense_resid_max'][i])}$ & "
                    f"{ps['dist_to_tube_mean'][i]:.4f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"Wrote {path}")
    with open(os.path.join(out, "manifest_61.json"), "w") as f:
        json.dump(man, f, indent=2)


# ----------------------------------------------------------------------------
# Stage 3: FIGURES -- all Section 6.1 figures from raw_data.npz.
# ----------------------------------------------------------------------------


def stage_figures():
    raw = R.load_raw(EXP)
    fig_dir = R.figures_dir()
    U_raw_ = raw["U_raw"]
    BU_raw = raw["BU_raw"]
    BU_extreme = raw["BU_extreme"]

    # ---- Figure: control_set ----------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(U_raw_[:, 0], U_raw_[:, 1], c="steelblue", s=50, zorder=3,
               label=rf"raw vertices ($K_{{\mathrm{{raw}}}}={len(U_raw_)}$)")
    ax.scatter(BU_extreme[:, 0], BU_extreme[:, 1], c="crimson", s=140,
               marker="*", zorder=4,
               label=rf"extreme vertices ($\tilde K={len(BU_extreme)}$)")
    hull_plt = scipy.spatial.ConvexHull(BU_raw)
    for s in hull_plt.simplices:
        ax.plot(BU_raw[s, 0], BU_raw[s, 1], "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$u_2$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "control_set.png"), dpi=180)
    plt.close(fig)

    # ---- Figure: loss_single ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(raw["loss_hist"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\mathcal{L}(\theta)$")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "loss_single.png"), dpi=180)
    plt.close(fig)

    # ---- Figure: trajectory_vs_tube ----------------------------------------
    t_ref = raw["t_ref"]
    tube = raw["tube"]
    x_pred = raw["x_pred_single"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for traj in tube[::5]:
        ax.plot(traj[:, 0], traj[:, 1], color="lightsteelblue", lw=0.6,
                alpha=0.5)
    ax.plot(tube[0, :, 0], tube[0, :, 1], color="steelblue", lw=1.2,
            label="reference (bang-bang)")
    ax.plot(x_pred[:, 0], x_pred[:, 1], "r-", lw=2.0, label=r"$x_\theta(t)$")
    ax.plot(*x0, "ko", ms=7, zorder=5, label=r"$x_0$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Phase portrait")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for traj in tube[::5]:
        ax.plot(t_ref, traj[:, 0], color="lightsteelblue", lw=0.5, alpha=0.4)
        ax.plot(t_ref, traj[:, 1], color="lightgreen", lw=0.5, alpha=0.4)
    ax.plot(t_ref, x_pred[:, 0], "r-", lw=2.0, label=r"$x_{\theta,1}(t)$")
    ax.plot(t_ref, x_pred[:, 1], "r--", lw=2.0, label=r"$x_{\theta,2}(t)$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("state")
    ax.set_title("State components vs time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "trajectory_vs_tube.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure: scalability (error bars) ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    ax = axes[0]
    ax.errorbar(raw["sc_K_values"], raw["sc_K_with_mean"],
                yerr=raw["sc_K_with_std"], fmt="o-", color="steelblue",
                capsize=3, label="with Quickhull")
    ax.errorbar(raw["sc_K_values"], raw["sc_K_without_mean"],
                yerr=raw["sc_K_without_std"], fmt="s--", color="crimson",
                capsize=3, label="without Quickhull")
    ax.set_xlabel(r"$K_{\mathrm{raw}}$")
    ax.set_ylabel("ms per epoch")
    ax.set_title(r"Training time vs $K$ ($d=2$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.errorbar(raw["sc_d_values"], raw["sc_d_with_mean"],
                yerr=raw["sc_d_with_std"], fmt="o-", color="steelblue",
                capsize=3, label="with Quickhull")
    ax.errorbar(raw["sc_d_values"], raw["sc_d_without_mean"],
                yerr=raw["sc_d_without_std"], fmt="s--", color="crimson",
                capsize=3, label="without Quickhull")
    ax.set_xlabel(r"$d$ (state dimension)")
    ax.set_ylabel("ms per epoch")
    ax.set_title(r"Training time vs $d$ ($K_{\mathrm{raw}}=32$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "scalability.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[figures] wrote 4 figures to {fig_dir}")


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "tables", "figures",
                                            "all"], default="all")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny configuration for a quick end-to-end test")
    args = parser.parse_args()

    if args.stage in ("train", "all"):
        params = R.load_config("exp61", smoke=args.smoke)
        cfg = Config(**params)
        if args.smoke:
            print(">>> SMOKE MODE: truncated budgets, "
                  "results not publication-grade.")
        stage_train(cfg, smoke=args.smoke)
    if args.stage in ("tables", "all"):
        stage_tables()
    if args.stage in ("figures", "all"):
        stage_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
