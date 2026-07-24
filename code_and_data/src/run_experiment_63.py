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

try:
    from paper_style import apply_paper_style

    apply_paper_style()
except ImportError:
    pass

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
# Export helpers
# ----------------------------------------------------------------------------


def sci_tex(x, digits=1):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp
    return rf"{mant:.{digits}f}\times10^{{{exp}}}"


def write_macros(path, macros):
    lines = [
        "%% AUTO-GENERATED by run_experiment_63.py -- do not edit by hand.",
        f"%% Generated: {datetime.now(timezone.utc).isoformat()}",
    ]
    for name, val in macros.items():
        lines.append(rf"\newcommand{{\{name}}}{{{val}}}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def hardware_string():
    cpu = platform.processor() or platform.machine()
    try:
        out = subprocess.run(["grep", "-m1", "model name", "/proc/cpuinfo"],
                             capture_output=True, text=True).stdout
        if ":" in out:
            cpu = out.split(":", 1)[1].strip()
    except Exception:
        pass
    gpus = tf.config.list_physical_devices("GPU")
    dev = f"GPU x{len(gpus)}" if gpus else "CPU"
    return f"{cpu} ({dev}), Python {platform.python_version()}"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_epochs = 40000
    if args.smoke:
        n_epochs = 200

    fig_dir = os.path.join("..", "results", "results_parabolic_case")
    gen_dir = os.path.join("..", "generated")
    ckpt_dir = os.path.join("..", "checkpoints")
    for d in (fig_dir, gen_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)
    t_run_start = time.time()

    # ---- 1. Reference solutions + Figure: reference_extinction_curves ----
    print("Reference solver sweep ...")
    ref = {}
    for lam in LAMBDA_SWEEP_REF:
        times, norms, u_fin, text, X, Y = run_reference_solver(lam)
        ref[lam] = dict(times=times, norms=norms, u_final=u_fin, text=text)
        st = f"t* = {text:.4f}" if text is not None else "not extinguished"
        print(f"  lambda = {lam}: {st}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for lam in LAMBDA_SWEEP_REF:
        r = ref[lam]
        label = rf"$\lambda={lam}$" + (" (heat eq.)" if lam == 0.0 else "")
        ax.semilogy(r["times"], np.maximum(r["norms"], 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2, label=label)
        if r["text"] is not None:
            ax.axvline(r["text"], color=LAMBDA_COLORS[lam], ls=":", lw=0.9,
                       alpha=0.8)
            ax.annotate(rf"$t^*={r['text']:.4f}$",
                        xy=(r["text"], NORM_FLOOR * 4),
                        xytext=(r["text"] + 0.004, NORM_FLOOR * 4),
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

    # ---- 2. Baseline training (lambda = 0.6, exact Phi) ------------------
    print(f"\nTraining baseline (lambda = {LAMBDA_BASELINE}, eps = 0) ...")
    net0, loss_hist0, val_hist0, val_pts0 = train_dr_pinn(
        lam=LAMBDA_BASELINE, T=T_HORIZON, n_epochs=n_epochs, seed=0)
    net0.save_weights(os.path.join(
        ckpt_dir, f"relay_lam{LAMBDA_BASELINE}.weights.h5"))  # (F3)
    val_loss0 = val_hist0[-1][1] if val_hist0 else loss_hist0[-1]

    # Training-narrative diagnostics (auto-detected, not hand-quoted):
    plateau_window = loss_hist0[100:min(4000, len(loss_hist0))]
    plateau_level = float(np.median(plateau_window)) if plateau_window else 0.0
    escape_epoch = next(
        (i for i, v in enumerate(loss_hist0)
         if plateau_level > 0 and i > 100 and v < plateau_level / 10),
        None)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(loss_hist0, color="lightsteelblue", lw=0.8,
                label="training loss (per step)", rasterized=True)
    ax.semilogy([e for e, _ in val_hist0], [v for _, v in val_hist0],
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(r0["times"], np.maximum(r0["norms"], 1e-16),
                     color="tab:blue", label="Reference solution", lw=2)
    axes[0].semilogy(times_net, np.maximum(l2_net, 1e-16), "--",
                     color="tab:orange", label="DR-PINN", lw=2)
    if r0["text"] is not None:
        axes[0].axvline(r0["text"], color="gray", ls=":", lw=0.9)
        axes[0].annotate(rf"$t^*_{{\rm ref}}={r0['text']:.4f}$",
                         xy=(r0["text"], NORM_FLOOR * 4),
                         xytext=(r0["text"] + 0.006, NORM_FLOOR * 4),
                         rotation=90, va="bottom", fontsize=10, color="0.3")
    axes[0].set_ylim(NORM_FLOOR, 2.0)
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$")
    axes[0].grid(True, which="major", alpha=0.25)
    axes[0].legend(loc="lower left")
    im = axes[1].imshow(np.abs(snaps_net[-1] - r0["u_final"]).T,
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

    # ---- 4. Branch-selection diagnostic (SAME net; F3) --------------------
    tf.random.set_seed(1)
    n_diag = 4000
    t_d = tf.random.uniform((n_diag, 1), 0.0, T_HORIZON)
    x_d = tf.random.uniform((n_diag, 1), 0.0, 1.0)
    y_d = tf.random.uniform((n_diag, 1), 0.0, 1.0)
    u_d, z_d = pde_operator(net0, t_d, x_d, y_d)
    s_vals = u_d.numpy().flatten()
    z_vals = z_d.numpy().flatten()
    z_front_max = float(np.max(np.abs(z_vals)))

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(s_vals, z_vals, s=5, alpha=0.20, color="tab:blue",
               edgecolors="none", label="Collocation points", rasterized=True)
    lam = LAMBDA_BASELINE
    s_min = min(s_vals.min(), -0.1)
    s_max = max(s_vals.max(), 0.1)
    ax.plot([0, s_max], [-lam, -lam], color="tab:red", lw=2,
            label=r"graph of $\Phi$", zorder=3)
    ax.plot([s_min, 0], [lam, lam], color="tab:red", lw=2, zorder=3)
    ax.plot([0, 0], [-lam, lam], color="tab:red", lw=2, zorder=3)
    ax.annotate(rf"$z=-\lambda={-lam}$", xy=(0.55, -lam),
                xytext=(0.55, -lam - 0.55), fontsize=11, color="tab:red",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.set_xlabel(r"$s = u_\theta(t,x,y)$")
    ax.set_ylabel(r"$z = \partial_t u_\theta - \Delta u_\theta$")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "branch_selection_diagnostic.png"),
                dpi=180)
    plt.close(fig)

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
        sweep[lam] = dict(times_net=times_l, l2_net=l2_l,
                          text_net=text_net, text_ref=ref[lam]["text"],
                          val_loss=val_l)
        print(f"  lambda={lam}: t*_ref = {ref[lam]['text']}, "
              f"t*_net = {text_net}, val loss = {val_l:.4e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for lam, res in sweep.items():
        ax.semilogy(ref[lam]["times"], np.maximum(ref[lam]["norms"], 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2,
                    label=rf"$\lambda={lam}$ (reference)")
        ax.semilogy(res["times_net"], np.maximum(res["l2_net"], 1e-16),
                    color=LAMBDA_COLORS[lam], lw=2, ls="--",
                    label=rf"$\lambda={lam}$ (DR-PINN)")
        if res["text_ref"] is not None:
            ax.axvline(res["text_ref"], color=LAMBDA_COLORS[lam], ls=":",
                       lw=0.9, alpha=0.7)
    ax.set_ylim(NORM_FLOOR, 2.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u(t)\|_{L^2(\Omega)}$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=10, ncol=2, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "extinction_time_sweep.png"), dpi=180)
    plt.close(fig)

    # ---- 7. Macros + manifest (F4) ----------------------------------------
    def relerr(lam):
        tr, tn = sweep[lam]["text_ref"], sweep[lam]["text_net"]
        if tr is None or tn is None:
            return None
        return abs(tn - tr) / tr

    macros = {
        # baseline
        "RelayValLoss": sci_tex(val_loss0),  # math-mode body
        "RelayRMS": f"{np.sqrt(val_loss0):.2f}",
        "RelayPlateauLevel": f"{plateau_level:.0f}",
        "RelayEscapeEpoch": (f"{escape_epoch:,}".replace(",", r"\,")
                             if escape_epoch else "--"),
        "RelayNormDisc": f"{norm_disc*100:.2f}\\%",
        "RelayMaxPtErr": sci_tex(max_pt_err),
        "RelayPostExtPlateau": rf"\approx{sci_tex(post_ext_plateau, 0)}",
        "RelayZFrontMax": f"{z_front_max:.0f}",
        "RelayTRefBase": f"{ref[LAMBDA_BASELINE]['text']:.4f}",
        # eps sensitivity (F2)
        "RelayEpsSens": (sci_tex(eps_max_rel)
                         if eps_max_rel > 0 else "0"),
    }
    for lam, key in zip(LAMBDA_SWEEP_MAIN, ["A", "B", "C"]):
        macros[f"RelayTRef{key}"] = f"{sweep[lam]['text_ref']:.4f}"
        macros[f"RelayTNet{key}"] = (f"{sweep[lam]['text_net']:.4f}"
                                     if sweep[lam]["text_net"] else "--")
        re_ = relerr(lam)
        macros[f"RelayRelErr{key}"] = (f"${re_*100:.1f}\\%$"
                                       if re_ is not None else "--")
    abs_errs = [abs(sweep[l]["text_net"] - sweep[l]["text_ref"])
                for l in LAMBDA_SWEEP_MAIN
                if sweep[l]["text_net"] and sweep[l]["text_ref"]]
    if abs_errs:
        macros["RelayAbsErrMin"] = f"{min(abs_errs):.3f}"
        macros["RelayAbsErrMax"] = f"{max(abs_errs):.3f}"
    write_macros(os.path.join(gen_dir, "results_63.tex"), macros)

    manifest = {
        "script": "run_experiment_63.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - t_run_start,
        "n_epochs": n_epochs,
        "eps_training": 0.0,
        "extinction_tolerance": EXTINCTION_TOL,
        "eval_time_step": T_HORIZON / (EVAL_N_TIMES - 1),
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__,
                     "scipy": scipy.__version__,
                     "tensorflow": tf.__version__},
        "hardware": hardware_string(),
        "reference_extinction_times": {str(l): ref[l]["text"]
                                       for l in LAMBDA_SWEEP_REF},
        "baseline": {"val_loss": val_loss0, "norm_disc": norm_disc,
                     "max_pt_err": max_pt_err,
                     "post_ext_plateau": post_ext_plateau,
                     "plateau_level": plateau_level,
                     "escape_epoch": escape_epoch,
                     "z_front_max": z_front_max},
        "eps_sensitivity": {"values": {str(k): v for k, v in eps_vals.items()},
                            "max_rel_dev": eps_max_rel},
        "sweep": {str(l): {"t_ref": sweep[l]["text_ref"],
                           "t_net": sweep[l]["text_net"],
                           "val_loss": sweep[l]["val_loss"]}
                  for l in LAMBDA_SWEEP_MAIN},
    }
    with open(os.path.join(gen_dir, "manifest_63.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest_63.json")
    print(f"\nDone in {time.time() - t_run_start:.0f} s.")


if __name__ == "__main__":
    sys.exit(main())
