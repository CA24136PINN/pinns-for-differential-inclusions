#!/usr/bin/env python3
"""Example 6.2 -- planar differential inclusion with a rotating ellipsoidal
control set: single-run replication pipeline.

Part A: hard-admissible selector optimization (24 nodal controls, RK4
        propagation, warm-started L-BFGS-B continuation over lambda_xi).
Part B: DR-PINN companion run (endpoint-hard ansatz, pure distance-residual
        loss with Newton projection onto the rotating ellipse and the exact
        envelope gradient obtained by holding the projected point constant).

One documented run produces, in a single execution:
  * all three paper figures of Section 6.2 + a companion diagnostic figure,
  * ../generated/results_62.tex  -- every number quoted in the text,
  * ../generated/manifest_62.json -- seeds, versions, hardware, all values,
  * ../checkpoints/ellipse_companion.weights.h5 -- the companion network.

Usage:  python run_experiment_62.py [--smoke]
"""

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
import scipy.optimize
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import repro_utils as R  # noqa: E402
from paper_style import apply_paper_style, PAPER_DPI  # noqa: E402

apply_paper_style()
tf.keras.backend.set_floatx("float64")

EXP = "exp62"

# ----------------------------------------------------------------------
# Problem data (Section 6.2)
# ----------------------------------------------------------------------
T = 3.0
X0 = np.array([0.0, 0.0])
XF = np.array([1.8, 0.1])

N_T = 120          # time-grid nodes t_i
N_C = 24           # control nodes (p^j, rho^j)
EPS_DIR = 1e-10    # normalization safeguard in d(t)

LAM_F = 2500.0
LAM_S = 1.0
LAM_C = 0.2
LAM_B = 1e-2
LAM_U = 5e-2
LAM_XI_STAGES = (1e-3, 1e-2, 1e-1)

SEED = 0


def axes_angle(t):
    """Semi-axes a(t), b(t) and rotation angle phi(t) of E(t)."""
    a = 0.3 + 0.4 * np.abs(np.sin(np.pi * t / T))
    b = 0.15 + 0.25 * np.abs(np.cos(np.pi * t / T))
    phi = np.pi * t / T
    return a, b, phi


def axes_angle_tf(t):
    a = 0.3 + 0.4 * tf.abs(tf.sin(np.pi * t / T))
    b = 0.15 + 0.25 * tf.abs(tf.cos(np.pi * t / T))
    phi = np.pi * t / T
    return a, b, phi


def drift_tf(t, x):
    """g(t, x) of eq. (ellipse-drift); x has shape (..., 2)."""
    g1 = tf.sin(x[..., 0]) + 0.15 * tf.cos(2.0 * np.pi * t / T)
    g2 = 0.5 * tf.cos(x[..., 1]) - 0.20 * tf.sin(2.0 * np.pi * t / T)
    return tf.stack([g1, g2], axis=-1)


def drift_np(t, x):
    g1 = np.sin(x[..., 0]) + 0.15 * np.cos(2.0 * np.pi * t / T)
    g2 = 0.5 * np.cos(x[..., 1]) - 0.20 * np.sin(2.0 * np.pi * t / T)
    return np.stack([g1, g2], axis=-1)


def level_np(t, xi):
    """Level function ell(t, xi); xi shape (..., 2), t broadcastable."""
    a, b, phi = axes_angle(t)
    c, s = np.cos(phi), np.sin(phi)
    u = c * xi[..., 0] + s * xi[..., 1]
    w = -s * xi[..., 0] + c * xi[..., 1]
    return (u / a) ** 2 + (w / b) ** 2


# ----------------------------------------------------------------------
# Part A -- hard-admissible selector optimization
# ----------------------------------------------------------------------
def build_interp_matrix(n_t, n_c):
    """Linear interpolation from n_c uniform nodes to n_t uniform grid."""
    t_grid = np.linspace(0.0, T, n_t)
    t_node = np.linspace(0.0, T, n_c)
    M = np.zeros((n_t, n_c))
    for i, t in enumerate(t_grid):
        j = min(np.searchsorted(t_node, t) - 1, n_c - 2)
        j = max(j, 0)
        w = (t - t_node[j]) / (t_node[j + 1] - t_node[j])
        M[i, j] = 1.0 - w
        M[i, j + 1] = w
    return t_grid, M


T_GRID, INTERP = build_interp_matrix(N_T, N_C)
DT = T / (N_T - 1)
INTERP_TF = tf.constant(INTERP, dtype=tf.float64)
T_GRID_TF = tf.constant(T_GRID, dtype=tf.float64)
A_GRID, B_GRID, PHI_GRID = axes_angle(T_GRID)
A_TF = tf.constant(A_GRID)
B_TF = tf.constant(B_GRID)
PHI_TF = tf.constant(PHI_GRID)


def selector_from_nodes_tf(z):
    """Map nodal variables z in R^{3 n_c} to (xi_i, ell_i) on the grid.

    xi(t_i) = R_phi ( s a d1, s b d2 ),  d = p/sqrt(|p|^2+eps),
    s = sigmoid(rho): interior of E(t_i) by construction.
    """
    p_nodes = tf.reshape(z[: 2 * N_C], (N_C, 2))
    r_nodes = z[2 * N_C:]
    p = tf.linalg.matmul(INTERP_TF, p_nodes)          # (n_t, 2)
    rho = tf.linalg.matvec(INTERP_TF, r_nodes)        # (n_t,)
    d = p / tf.sqrt(tf.reduce_sum(p ** 2, axis=1, keepdims=True) + EPS_DIR)
    s = tf.sigmoid(rho)
    u = s * A_TF * d[:, 0]
    w = s * B_TF * d[:, 1]
    c, sn = tf.cos(PHI_TF), tf.sin(PHI_TF)
    xi = tf.stack([c * u - sn * w, sn * u + c * w], axis=1)
    ell = s ** 2 * tf.reduce_sum(d ** 2, axis=1)
    return xi, ell


def rk4_rollout_tf(xi):
    """RK4 propagation of x' = g(t,x) + xi(t); xi linear between nodes."""
    x = tf.constant(X0, dtype=tf.float64)
    xs = tf.TensorArray(tf.float64, size=N_T)
    xs = xs.write(0, x)
    for i in tf.range(N_T - 1):
        t0 = T_GRID_TF[i]
        xi0 = xi[i]
        xi1 = xi[i + 1]
        xim = 0.5 * (xi0 + xi1)
        k1 = drift_tf(t0, x) + xi0
        k2 = drift_tf(t0 + 0.5 * DT, x + 0.5 * DT * k1) + xim
        k3 = drift_tf(t0 + 0.5 * DT, x + 0.5 * DT * k2) + xim
        k4 = drift_tf(t0 + DT, x + DT * k3) + xi1
        x = x + DT / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        xs = xs.write(i + 1, x)
    return xs.stack()


@tf.function
def objective_tf(z, lam_xi):
    xi, ell = selector_from_nodes_tf(z)
    xs = rk4_rollout_tf(xi)
    mism = tf.reduce_sum((xs[-1] - XF) ** 2)
    j_xi = tf.reduce_mean(ell)
    d1 = xi[1:] - xi[:-1]
    d2 = xi[2:] - 2.0 * xi[1:-1] + xi[:-2]
    j_s = tf.reduce_mean(tf.reduce_sum(d1 ** 2, axis=1))
    j_c = tf.reduce_mean(tf.reduce_sum(d2 ** 2, axis=1))
    j_b = tf.reduce_mean(ell ** 3)
    zn = tf.reshape(z, (3, N_C))
    j_u = tf.reduce_mean((zn[:, 1:] - zn[:, :-1]) ** 2)
    return (LAM_F * mism + lam_xi * j_xi + LAM_S * j_s + LAM_C * j_c
            + LAM_B * j_b + LAM_U * j_u)


@tf.function
def objective_grad_tf(z, lam_xi):
    with tf.GradientTape() as tape:
        tape.watch(z)
        val = objective_tf(z, lam_xi)
    return val, tape.gradient(val, z)


def run_part_a(maxiter):
    rng = np.random.default_rng(SEED)
    z = rng.normal(0.0, 0.5, size=3 * N_C)

    def make_fun(lam_xi):
        lam = tf.constant(lam_xi, dtype=tf.float64)

        def fun(zv):
            v, g = objective_grad_tf(tf.constant(zv, dtype=tf.float64), lam)
            return float(v.numpy()), g.numpy()

        return fun

    stages = []
    for lam_xi in LAM_XI_STAGES:
        res = scipy.optimize.minimize(
            make_fun(lam_xi), z, jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter, "maxfun": 4 * maxiter,
                     "ftol": 1e-16, "gtol": 1e-12},
        )
        z = res.x
        xi, ell = selector_from_nodes_tf(tf.constant(z))
        xs = rk4_rollout_tf(xi)
        mism = float(np.linalg.norm(xs.numpy()[-1] - XF))
        stages.append({
            "lam_xi": lam_xi,
            "mismatch": mism,
            "mean_level": float(np.mean(ell.numpy())),
            "max_level": float(np.max(ell.numpy())),
            "nit": int(res.nit),
        })
        print(f"  stage lam_xi={lam_xi:g}: |x(T)-xf| = {mism:.2e}, "
              f"mean l = {stages[-1]['mean_level']:.3f}, "
              f"max l = {stages[-1]['max_level']:.3f}  ({res.nit} its)")
    return z, stages


def dense_admissibility_check(z, n_dense):
    """A-posteriori check: linear interpolation of xi between grid nodes,
    level evaluated against the *intermediate* ellipse E(t)."""
    xi, _ = selector_from_nodes_tf(tf.constant(z))
    xi = xi.numpy()
    t_dense = np.linspace(0.0, T, n_dense)
    xi_dense = np.stack([np.interp(t_dense, T_GRID, xi[:, k])
                         for k in range(2)], axis=1)
    lev = level_np(t_dense, xi_dense)
    return float(np.max(lev)), int(np.sum(lev > 1.0))


def extremal_down_probe():
    """Selector taking the support point of E(t) in direction -e2 at every
    node; RK4 rollout reports the most negative reachable x2(T)."""
    q = np.array([0.0, -1.0])
    xi = np.zeros((N_T, 2))
    for i, t in enumerate(T_GRID):
        a, b, phi = axes_angle(t)
        c, s = np.cos(phi), np.sin(phi)
        q_loc = np.array([c * q[0] + s * q[1], -s * q[0] + c * q[1]])
        v_loc = np.array([a ** 2 * q_loc[0], b ** 2 * q_loc[1]])
        v_loc /= np.sqrt((a * q_loc[0]) ** 2 + (b * q_loc[1]) ** 2)
        xi[i] = np.array([c * v_loc[0] - s * v_loc[1],
                          s * v_loc[0] + c * v_loc[1]])
    x = X0.copy()
    for i in range(N_T - 1):
        t0 = T_GRID[i]
        xi0, xi1 = xi[i], xi[i + 1]
        xim = 0.5 * (xi0 + xi1)
        k1 = drift_np(t0, x) + xi0
        k2 = drift_np(t0 + 0.5 * DT, x + 0.5 * DT * k1) + xim
        k3 = drift_np(t0 + 0.5 * DT, x + 0.5 * DT * k2) + xim
        k4 = drift_np(t0 + DT, x + DT * k3) + xi1
        x = x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return float(x[1])


# ----------------------------------------------------------------------
# Part B -- DR-PINN companion run
# ----------------------------------------------------------------------
NEWTON_ITERS = 30


def project_ellipse_tf(v, t):
    """Projection of v (batch, 2) onto E(t) (batch,), local-frame Newton on
    the secular equation sum_k (sigma_k v_k / (sigma_k^2 + mu))^2 = 1."""
    a, b, phi = axes_angle_tf(t)
    c, s = tf.cos(phi), tf.sin(phi)
    u = c * v[:, 0] + s * v[:, 1]
    w = -s * v[:, 0] + c * v[:, 1]
    lev = (u / a) ** 2 + (w / b) ** 2

    mu = tf.zeros_like(lev)
    for _ in range(NEWTON_ITERS):
        f = (a * u / (a ** 2 + mu)) ** 2 + (b * w / (b ** 2 + mu)) ** 2 - 1.0
        df = (-2.0 * (a * u) ** 2 / (a ** 2 + mu) ** 3
              - 2.0 * (b * w) ** 2 / (b ** 2 + mu) ** 3)
        mu = tf.maximum(mu - f / df, 0.0)

    pu = a ** 2 * u / (a ** 2 + mu)
    pw = b ** 2 * w / (b ** 2 + mu)
    proj = tf.stack([c * pu - s * pw, s * pu + c * pw], axis=1)
    inside = lev <= 1.0
    proj = tf.where(inside[:, None], v, proj)
    return proj, lev, inside


def dist2_tf(v, t):
    """Squared inclusion distance with the exact envelope gradient: the
    entire projected point is held constant in the backward pass."""
    proj, lev, inside = project_ellipse_tf(v, t)
    diff = v - tf.stop_gradient(proj)
    d2 = tf.reduce_sum(diff ** 2, axis=1)
    return tf.where(inside, tf.zeros_like(d2), d2), lev


def build_companion_net():
    tf.random.set_seed(SEED)
    net = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(1,))]
        + [tf.keras.layers.Dense(64, activation="tanh") for _ in range(3)]
        + [tf.keras.layers.Dense(2)]
    )
    return net


def companion_state_and_velocity(net, t):
    """x_theta(t) = x0 + t/T (xf - x0) + t(T-t) N(t) and its derivative."""
    t = tf.reshape(t, (-1, 1))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        n = net(t)
        x = (X0[None, :] + t / T * (XF - X0)[None, :]
             + t * (T - t) * n)
    dx = tape.batch_jacobian(x, t)[:, :, 0]
    del tape
    return x, dx


def run_part_b(n_epochs, lr_hold, lr_decay_every, lr_decay_rate,
               n_col=512, n_eval=2000):
    net = build_companion_net()
    t_col = tf.constant(np.linspace(0.0, T, n_col), dtype=tf.float64)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        2e-3, decay_steps=lr_decay_every, decay_rate=lr_decay_rate,
        staircase=True)

    def lr_fn(step):
        return 2e-3 if step < lr_hold else float(lr(step - lr_hold))

    opt = tf.keras.optimizers.Adam(learning_rate=2e-3)

    @tf.function
    def loss_fn():
        x, dx = companion_state_and_velocity(net, t_col)
        v = dx - drift_tf(t_col, x)
        d2, _ = dist2_tf(v, t_col)
        return tf.reduce_mean(d2)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(grads, net.trainable_variables))
        return loss

    history = []
    for epoch in range(n_epochs):
        opt.learning_rate.assign(lr_fn(epoch))
        loss = float(train_step().numpy())
        history.append((epoch, loss, float(opt.learning_rate.numpy())))
        if epoch % max(1, n_epochs // 10) == 0:
            print(f"  epoch {epoch:6d} | loss {loss:.4e}")
    final_loss = float(loss_fn().numpy())

    # dense-grid evaluation
    t_dense = tf.constant(np.linspace(0.0, T, n_eval), dtype=tf.float64)
    x, dx = companion_state_and_velocity(net, t_dense)
    v = dx - drift_tf(t_dense, x)
    d2, lev = dist2_tf(v, t_dense)
    dist = np.sqrt(np.maximum(d2.numpy(), 0.0))
    metrics = {
        "final_collocation_loss": final_loss,
        "distance_mean_dense": float(np.mean(dist)),
        "distance_max_dense": float(np.max(dist)),
        "distance_rms_dense": float(np.sqrt(np.mean(dist ** 2))),
        "level_mean_dense": float(np.mean(lev.numpy())),
        "level_max_dense": float(np.max(lev.numpy())),
        "inside_fraction_dense": float(np.mean(lev.numpy() <= 1.0)),
        "initial_endpoint_error": float(np.linalg.norm(
            x.numpy()[0] - X0)),
        "terminal_endpoint_error": float(np.linalg.norm(
            x.numpy()[-1] - XF)),
    }

    # envelope-gradient finite-difference check at an exterior test point
    t_test = tf.constant([0.7], dtype=tf.float64)
    v_test = tf.Variable([[1.7, -1.3]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        d2t, _ = dist2_tf(v_test, t_test)
    g_auto = tape.gradient(d2t, v_test).numpy()[0]
    h = 1e-6
    g_fd = np.zeros(2)
    for k in range(2):
        vp = v_test.numpy().copy(); vp[0, k] += h
        vm = v_test.numpy().copy(); vm[0, k] -= h
        dp, _ = dist2_tf(tf.constant(vp), t_test)
        dm, _ = dist2_tf(tf.constant(vm), t_test)
        g_fd[k] = (float(dp.numpy()[0]) - float(dm.numpy()[0])) / (2 * h)
    metrics["gradient_check_relative_error"] = float(
        np.linalg.norm(g_auto - g_fd) / np.linalg.norm(g_fd))
    metrics["gradient_check_autograd"] = g_auto.tolist()
    metrics["gradient_check_fd"] = g_fd.tolist()

    net.save_weights(os.path.join(R.checkpoints_dir(EXP),
                                  "ellipse_companion.weights.h5"))
    with open(os.path.join(R.raw_dir(EXP), "companion_history.csv"), "w") as f:
        f.write("epoch,inclusion_loss,learning_rate\n")
        for e, l, r in history:
            f.write(f"{e},{l},{r}\n")
    return net, metrics, history, (t_dense.numpy(), lev.numpy())


# ----------------------------------------------------------------------
# Figures (paper style)
# ----------------------------------------------------------------------
def ellipse_ring(t, n=100):
    a, b, phi = axes_angle(t)
    th = np.linspace(0, 2 * np.pi, n)
    u, w = a * np.cos(th), b * np.sin(th)
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * u - s * w, s * u + c * w], axis=1)


def fig_velocity_tube(xi, xs):
    g = drift_np(T_GRID, xs)
    vel = g + xi

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    th = np.linspace(0, 2 * np.pi, 60)
    Vx = np.zeros((N_T, th.size))
    Vy = np.zeros((N_T, th.size))
    Tt = np.tile(T_GRID[:, None], (1, th.size))
    for i, t in enumerate(T_GRID):
        ring = ellipse_ring(t, th.size)
        Vx[i] = g[i, 0] + ring[:, 0]
        Vy[i] = g[i, 1] + ring[:, 1]
    ax.plot_surface(Vx, Vy, Tt, alpha=0.22, color="tab:blue",
                    linewidth=0, antialiased=True)
    ax.plot(g[:, 0], g[:, 1], T_GRID, color="tab:blue", lw=2,
            label=r"drift $g(t,x_z(t))$")
    ax.plot(vel[:, 0], vel[:, 1], T_GRID, color="tab:orange", lw=2,
            label=r"selected velocity $\dot{x}_z(t)$")
    for i in range(0, N_T, 6):
        ax.plot([g[i, 0], vel[i, 0]], [g[i, 1], vel[i, 1]],
                [T_GRID[i], T_GRID[i]], color="0.5", lw=0.8)
    ax.set_xlabel(r"$v_1$"); ax.set_ylabel(r"$v_2$"); ax.set_zlabel(r"$t$")
    ax.legend(loc="upper left")
    ax.view_init(elev=18, azim=-60)
    fig.savefig(os.path.join(R.figures_dir(), "ellipse_velocity_tube.png"),
                dpi=PAPER_DPI)
    plt.close(fig)


def fig_selector_level(xi, ell):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for i in range(0, N_T, 6):
        ring = ellipse_ring(T_GRID[i])
        ax.plot(ring[:, 0], ring[:, 1], color="tab:blue", lw=0.5,
                alpha=0.45)
    ax.plot(xi[:, 0], xi[:, 1], color="0.4", lw=0.9, zorder=2)
    sc = ax.scatter(xi[:, 0], xi[:, 1], c=ell, cmap="viridis", s=22,
                    vmin=0.0, vmax=1.0, zorder=3)
    fig.colorbar(sc, ax=ax, label=r"level $\ell(t_i,\xi_z(t_i))$")
    ax.set_xlabel(r"$\xi_1$"); ax.set_ylabel(r"$\xi_2$")
    ax.set_aspect("equal")
    fig.savefig(os.path.join(R.figures_dir(), "ellipse_selector_level.png"),
                dpi=PAPER_DPI)
    plt.close(fig)


def fig_state_trajectory(xs, ell):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(xs[:, 0], xs[:, 1], color="tab:blue", lw=2)
    ax.scatter(*X0, color="tab:green", zorder=3, label=r"$x_0$")
    ax.scatter(*XF, color="tab:red", marker="*", s=140, zorder=3,
               label=r"$x_{\rm f}$")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.legend()
    ax = axes[1]
    ax.plot(T_GRID, ell, color="tab:blue", lw=2)
    ax.axhline(1.0, color="0.3", ls="--", lw=1)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\ell(t,\xi_z(t))$")
    ax.set_ylim(0.0, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(R.figures_dir(), "ellipse_state_trajectory.png"),
                dpi=PAPER_DPI)
    plt.close(fig)


def fig_companion(history, dense_levels):
    ep = [h[0] for h in history]
    lo = [h[1] for h in history]
    t_dense, lev = dense_levels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(ep, lo, color="tab:blue", lw=1.2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(r"collocation loss $\mathcal{L}(\theta)$")
    axes[1].plot(t_dense, lev, color="tab:blue", lw=1.5)
    axes[1].axhline(1.0, color="0.3", ls="--", lw=1)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"level $\ell(t,\xi_\theta(t))$")
    fig.tight_layout()
    fig.savefig(os.path.join(R.figures_dir(), "companion_diagnostics.png"),
                dpi=PAPER_DPI)
    plt.close(fig)


# ----------------------------------------------------------------------
# Stage 1: TRAIN -- Part A + Part B; writes results/raw/exp62/
# (raw_data.npz + manifest_62.json + checkpoints/).  NO figures here.
# ----------------------------------------------------------------------


def stage_train(params, smoke=False):
    maxiter = int(params["maxiter"])
    epochs = int(params["epochs"])
    n_dense = int(params["n_dense"])
    lr_hold = int(params["lr_hold"])
    dec_every = int(params["lr_decay_every"])
    dec_rate = float(params["lr_decay_rate"])

    t_start = time.time()
    print("Part A: hard-admissible selector optimization "
          "(L-BFGS-B continuation) ...")
    z, stages = run_part_a(maxiter)
    dense_max_l, n_viol = dense_admissibility_check(z, n_dense)
    print(f"  dense a-posteriori check ({n_dense} pts): "
          f"max l = {dense_max_l:.3f}, violations = {n_viol}")
    x2_ext = extremal_down_probe()
    print(f"  extremal -e2 probe: x2(T) = {x2_ext:.3f}")

    # Precompute every array any Part-A figure needs (pure numpy afterwards)
    xi_tf, ell_tf = selector_from_nodes_tf(tf.constant(z))
    xi = xi_tf.numpy()
    ell = ell_tf.numpy()
    xs = rk4_rollout_tf(xi_tf).numpy()

    print("Part B: DR-PINN companion run ...")
    net, comp, history, dense_levels = run_part_b(
        epochs, lr_hold, dec_every, dec_rate)
    print(f"  final loss = {comp['final_collocation_loss']:.2e}, "
          f"dense dist mean/max = {comp['distance_mean_dense']:.2e}"
          f"/{comp['distance_max_dense']:.2e}, "
          f"grad check = {comp['gradient_check_relative_error']:.2e}")
    t_dense, lev = dense_levels

    R.save_raw(
        EXP,
        z=z, xi=xi, ell=ell, xs=xs,
        hist_epochs=np.asarray([h[0] for h in history]),
        hist_loss=np.asarray([h[1] for h in history]),
        hist_lr=np.asarray([h[2] for h in history]),
        companion_t_dense=t_dense,
        companion_level_dense=lev,
    )

    manifest = {
        "script": "run_experiment_62.py",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_seconds": time.time() - t_start,
        "smoke": bool(smoke),
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tensorflow": tf.__version__,
        },
        "hardware": R.hardware_string(),
        "config": {
            "T": T, "x0": X0.tolist(), "xf": XF.tolist(),
            "n_t": N_T, "n_c": N_C,
            "weights": {"lam_f": LAM_F, "lam_s": LAM_S, "lam_c": LAM_C,
                        "lam_b": LAM_B, "lam_u": LAM_U,
                        "lam_xi_stages": list(LAM_XI_STAGES)},
            "newton_iterations": NEWTON_ITERS,
            "maxiter": maxiter, "epochs": epochs, "n_dense": n_dense,
        },
        "part_a": {"stages": stages,
                   "dense_max_level": dense_max_l,
                   "dense_violations": n_viol,
                   "n_dense": n_dense,
                   "extremal_down_x2T": x2_ext},
        "part_b": comp,
    }
    R.save_manifest(EXP, manifest, "62")
    print(f"\n[train] done in {time.time() - t_start:.0f} s.")


# ----------------------------------------------------------------------
# Stage 2: TABLES -- results/aggregated/results_62.tex from the manifest.
# ----------------------------------------------------------------------


def stage_tables():
    man = R.load_manifest(EXP, "62")
    stages = man["part_a"]["stages"]
    comp = man["part_b"]
    n_dense = man["part_a"]["n_dense"]

    macros = {
        "EllipMismA": R.sci_tex(stages[0]["mismatch"]),
        "EllipMeanLA": f"{stages[0]['mean_level']:.3f}",
        "EllipMaxLA": f"{stages[0]['max_level']:.3f}",
        "EllipMismB": R.sci_tex(stages[1]["mismatch"]),
        "EllipMeanLB": f"{stages[1]['mean_level']:.3f}",
        "EllipMaxLB": f"{stages[1]['max_level']:.3f}",
        "EllipMismC": R.sci_tex(stages[2]["mismatch"]),
        "EllipMeanLC": f"{stages[2]['mean_level']:.3f}",
        "EllipMaxLC": f"{stages[2]['max_level']:.3f}",
        "EllipDenseMaxL": f"{man['part_a']['dense_max_level']:.3f}",
        "EllipNDense": rf"{n_dense // 1000}\,{n_dense % 1000:03d}"
        if n_dense >= 1000 else str(n_dense),
        "EllipExtremalXtwo": f"{man['part_a']['extremal_down_x2T']:.2f}",
        "EllipCompLoss": R.sci_tex(comp["final_collocation_loss"]),
        "EllipCompDenseMean": R.sci_tex(comp["distance_mean_dense"]),
        "EllipCompDenseMax": R.sci_tex(comp["distance_max_dense"]),
        "EllipCompMeanL": f"{comp['level_mean_dense']:.3f}",
        "EllipCompMaxL": f"{comp['level_max_dense']:.6f}",
        "EllipCompGradCheck": R.sci_tex(
            comp["gradient_check_relative_error"]),
    }
    out = R.aggregated_dir()
    R.write_macros(os.path.join(out, "results_62.tex"), macros,
                   "run_experiment_62.py --stage tables")
    with open(os.path.join(out, "manifest_62.json"), "w") as f:
        json.dump(man, f, indent=2)


# ----------------------------------------------------------------------
# Stage 3: FIGURES -- all Section 6.2 figures from raw_data.npz.
# ----------------------------------------------------------------------


def stage_figures():
    raw = R.load_raw(EXP)
    fig_velocity_tube(raw["xi"], raw["xs"])
    fig_selector_level(raw["xi"], raw["ell"])
    fig_state_trajectory(raw["xs"], raw["ell"])
    history = list(zip(raw["hist_epochs"], raw["hist_loss"], raw["hist_lr"]))
    fig_companion(history,
                  (raw["companion_t_dense"], raw["companion_level_dense"]))
    print(f"[figures] wrote 4 figures to {R.figures_dir()}")


# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "tables", "figures", "all"],
                    default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end test (small iteration budgets)")
    args = ap.parse_args()

    if args.stage in ("train", "all"):
        params = R.load_config("exp62", smoke=args.smoke)
        if args.smoke:
            print(">>> SMOKE MODE: truncated budgets, "
                  "results not publication-grade.")
        stage_train(params, smoke=args.smoke)
    if args.stage in ("tables", "all"):
        stage_tables()
    if args.stage in ("figures", "all"):
        stage_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
