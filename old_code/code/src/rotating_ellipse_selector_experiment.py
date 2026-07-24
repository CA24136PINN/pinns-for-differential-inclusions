#!/usr/bin/env python3
"""Rotating-ellipse selector-steering experiment (Section 6.2 of the manuscript).

Solves, exactly as specified in notes/notes_convex/rotating_ellipse_inclusion_convex.tex,
the finite-dimensional steering problem for the planar differential inclusion

    xdot(t) in g(t, x(t)) + E(t),   x(0) = x0,

with the rotating ellipsoidal control set E(t), by the hard-admissible selector
parametrization (K control nodes, direction normalization + sigmoid radius),
RK4 time integration, and L-BFGS-B with warm-started continuation over the
selector-regularization weight lambda_xi in {1e-3, 1e-2, 1e-1}.

Outputs (paper style, dpi=180):
    ellipse_velocity_tube.png
    ellipse_selector_level.png
    ellipse_state_trajectory.png
and prints the quantitative results quoted in Section 6.2.
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply_paper_style

# ----------------------------------------------------------------------------
# Problem data (verbatim from the notes)
# ----------------------------------------------------------------------------
T = 3.0
N = 120                      # time-grid points
K = 24                       # control nodes
X0 = np.array([0.0, 0.0])
# Target state x_f. NOTE: the draft notes proposed (1.0, -0.2), but that
# point is NOT reachable for this drift/ellipse/T: even the extremal selector
# minimizing x2 (support point of E(t) in direction -e2 at every instant)
# only attains x2(T) ~= -0.05, because the drift g2 = 0.5 cos(x2) - 0.2 sin(.)
# pushes upward with magnitude ~0.5 near x2 = 0 while the ellipse's vertical
# reach is 0.15--0.7. The target below lies inside the reachable set but
# close to its boundary: steering to it forces the selector to operate near
# the boundary of E(t) for most of the horizon (mean level ~0.9).
XF = np.array([1.8, 0.1])
EPS = 1e-10                  # direction-normalization regularization

LAM_T = 2500.0               # terminal weight
LAM_S = 1.0                  # first-difference smoothness
LAM_C = 0.2                  # second-difference curvature
LAM_B = 1e-2                 # cubic soft barrier
LAM_U = 5e-2                 # nodal smoothness
LAM_XI_CONTINUATION = [1e-3, 1e-2, 1e-1]

t_grid = np.linspace(0.0, T, N)          # t_i = i*Dt, Dt = T/(N-1)
DT = t_grid[1] - t_grid[0]
node_grid = np.linspace(0.0, T, K)

OUT_DIR = os.environ.get("ELLIPSE_OUT_DIR",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "results", "results_rotating_ellipse"))


def axes_ab(t):
    a = 0.30 + 0.40 * np.abs(np.sin(np.pi * t / T))
    b = 0.15 + 0.25 * np.abs(np.cos(np.pi * t / T))
    return a, b


def angle(t):
    return np.pi * t / T


def drift(t, x):
    """g(t, x) from the notes: nonlinear in the state, periodic in time."""
    x1, x2 = x[..., 0], x[..., 1]
    g1 = np.sin(x1) + 0.15 * np.cos(2.0 * np.pi * t / T)
    g2 = 0.5 * np.cos(x2) - 0.20 * np.sin(2.0 * np.pi * t / T)
    return np.stack([g1, g2], axis=-1)


def selector_from_z(z):
    """Nodal variables z = (p^0..p^{K-1}, rho^0..rho^{K-1}) -> xi(t_i), level(t_i).

    Accepts a single z of shape (3K,) or a batch of shape (M, 3K); returns
    xi of shape (..., N, 2) and level of shape (..., N).

    p^j in R^2 are direction nodes, rho^j in R radius nodes; linear
    interpolation to the fine grid, then
        d = p / sqrt(|p|^2 + eps),  s = sigmoid(rho),
        (u, w) = (s a d1, s b d2),  xi = R_phi (u, w)^T,
    which lies in the open ellipse E(t) by construction.
    """
    z = np.atleast_2d(z)                       # (M, 3K)
    M = z.shape[0]
    p_nodes = z[:, :2 * K].reshape(M, K, 2)
    rho_nodes = z[:, 2 * K:]                   # (M, K)

    # Linear interpolation nodes -> fine grid, precomputed weights
    idx = np.searchsorted(node_grid, t_grid, side="right") - 1
    idx = np.clip(idx, 0, K - 2)
    lam = (t_grid - node_grid[idx]) / (node_grid[idx + 1] - node_grid[idx])
    p = ((1 - lam)[None, :, None] * p_nodes[:, idx, :]
         + lam[None, :, None] * p_nodes[:, idx + 1, :])       # (M, N, 2)
    rho = (1 - lam)[None, :] * rho_nodes[:, idx] + lam[None, :] * rho_nodes[:, idx + 1]

    d = p / np.sqrt(np.sum(p**2, axis=-1, keepdims=True) + EPS)
    s = 1.0 / (1.0 + np.exp(-rho))             # (M, N)
    a, b = axes_ab(t_grid)                     # (N,)
    u = s * a[None, :] * d[..., 0]
    w = s * b[None, :] * d[..., 1]
    phi = angle(t_grid)
    xi = np.stack([np.cos(phi)[None, :] * u - np.sin(phi)[None, :] * w,
                   np.sin(phi)[None, :] * u + np.cos(phi)[None, :] * w], axis=-1)
    level = (u / a[None, :]) ** 2 + (w / b[None, :]) ** 2     # = s^2|d|^2 < 1
    if M == 1:
        return xi[0], level[0]
    return xi, level


def drift_batch(t, x):
    """g(t, x) for x of shape (M, 2), scalar t."""
    g1 = np.sin(x[:, 0]) + 0.15 * np.cos(2.0 * np.pi * t / T)
    g2 = 0.5 * np.cos(x[:, 1]) - 0.20 * np.sin(2.0 * np.pi * t / T)
    return np.stack([g1, g2], axis=-1)


def integrate_rk4(xi):
    """Classical RK4 for xdot = g(t,x) + xi(t); xi of shape (..., N, 2).

    Linear interpolation of xi between grid points means the midpoint value
    is exactly the average of consecutive grid values, which is precomputed.
    """
    single = (xi.ndim == 2)
    Xi = xi[None] if single else xi            # (M, N, 2)
    M = Xi.shape[0]
    xi_mid = 0.5 * (Xi[:, :-1] + Xi[:, 1:])    # (M, N-1, 2)

    x = np.zeros((M, N, 2))
    x[:, 0] = X0
    for k in range(N - 1):
        tk = t_grid[k]
        xk = x[:, k]
        k1 = drift_batch(tk, xk) + Xi[:, k]
        k2 = drift_batch(tk + DT / 2, xk + DT / 2 * k1) + xi_mid[:, k]
        k3 = drift_batch(tk + DT / 2, xk + DT / 2 * k2) + xi_mid[:, k]
        k4 = drift_batch(tk + DT, xk + DT * k3) + Xi[:, k + 1]
        x[:, k + 1] = xk + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x[0] if single else x


def cost_batch(Z, lam_xi):
    """Cost for a batch Z of shape (M, 3K); returns shape (M,)."""
    Z = np.atleast_2d(Z)
    M = Z.shape[0]
    xi, level = selector_from_z(Z)
    if M == 1:
        xi, level = xi[None], level[None]
    x = integrate_rk4(xi)
    if x.ndim == 2:
        x = x[None]

    J_T = np.sum((x[:, -1] - XF[None, :]) ** 2, axis=-1)
    J_xi = np.mean(level, axis=-1)
    J_s = np.mean(np.sum(np.diff(xi, axis=1) ** 2, axis=-1), axis=-1)
    J_c = np.mean(np.sum((xi[:, 2:] - 2 * xi[:, 1:-1] + xi[:, :-2]) ** 2,
                         axis=-1), axis=-1)
    J_b = np.mean(level ** 3, axis=-1)
    p_nodes = Z[:, :2 * K].reshape(M, K, 2)
    rho_nodes = Z[:, 2 * K:]
    J_u = (np.mean(np.sum(np.diff(p_nodes, axis=1) ** 2, axis=-1), axis=-1)
           + np.mean(np.diff(rho_nodes, axis=1) ** 2, axis=-1))

    return (LAM_T * J_T + lam_xi * J_xi + LAM_S * J_s + LAM_C * J_c
            + LAM_B * J_b + LAM_U * J_u)


def cost(z, lam_xi):
    return float(cost_batch(z, lam_xi)[0])


_FD_H = 1e-6

def cost_grad(z, lam_xi):
    """Central-difference gradient, all perturbations in one vectorized batch."""
    n = z.size
    Z = np.repeat(z[None, :], 2 * n, axis=0)
    Z[:n, :] += _FD_H * np.eye(n)
    Z[n:, :] -= _FD_H * np.eye(n)
    c = cost_batch(Z, lam_xi)
    return (c[:n] - c[n:]) / (2 * _FD_H)


def solve():
    rng = np.random.default_rng(0)
    z = 0.1 * rng.standard_normal(3 * K)
    history = []
    for lam_xi in LAM_XI_CONTINUATION:
        res = minimize(cost, z, args=(lam_xi,), jac=cost_grad,
                       method="L-BFGS-B",
                       options=dict(maxiter=500, maxfun=200000))
        z = res.x
        xi, level = selector_from_z(z)
        x = integrate_rk4(xi)
        err = float(np.linalg.norm(x[-1] - XF))
        history.append(dict(lam_xi=lam_xi, cost=float(res.fun),
                            terminal_error=err,
                            mean_level=float(level.mean()),
                            max_level=float(level.max()),
                            nit=res.nit))
        print(f"lambda_xi = {lam_xi:8.0e}: |x(T)-x_f| = {err:.3e}   "
              f"mean l = {level.mean():.3f}   max l = {level.max():.3f}   "
              f"(L-BFGS-B iters: {res.nit})")
    return z, history


# ----------------------------------------------------------------------------
# Figures (paper style)
# ----------------------------------------------------------------------------

def ellipse_boundary(ti, n_phi=100, center=None):
    a, b = axes_ab(ti)
    phi = angle(ti)
    tt = np.linspace(0, 2 * np.pi, n_phi)
    ex, ey = a * np.cos(tt), b * np.sin(tt)
    rx = np.cos(phi) * ex - np.sin(phi) * ey
    ry = np.sin(phi) * ex + np.cos(phi) * ey
    if center is not None:
        rx, ry = rx + center[0], ry + center[1]
    return rx, ry


def make_figures(z):
    xi, level = selector_from_z(z)
    x = integrate_rk4(xi)
    gc = drift(t_grid, x)          # centerline g(t, x(t))
    v = gc + xi                    # selected velocity

    os.makedirs(OUT_DIR, exist_ok=True)

    # -- Figure 1: admissible velocity tube in (v1, v2, t) space ------------
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(projection="3d")
    n_phi = 60
    tt = np.linspace(0, 2 * np.pi, n_phi)
    Xs = np.zeros((N, n_phi)); Ys = np.zeros((N, n_phi)); Zs = np.zeros((N, n_phi))
    for i, ti in enumerate(t_grid):
        rx, ry = ellipse_boundary(ti, n_phi, center=gc[i])
        Xs[i], Ys[i], Zs[i] = rx, ry, ti
    ax.plot_surface(Xs, Ys, Zs, alpha=0.22, color="tab:blue",
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax.plot(gc[:, 0], gc[:, 1], t_grid, color="tab:blue", lw=2,
            label=r"centerline $g(t,x(t))$")
    ax.plot(v[:, 0], v[:, 1], t_grid, color="tab:orange", lw=2,
            label=r"selected velocity $\dot{x}(t)$")
    for i in range(0, N, 10):
        ax.plot([gc[i, 0], v[i, 0]], [gc[i, 1], v[i, 1]], [t_grid[i]] * 2,
                color="0.45", lw=0.7, alpha=0.7)
    ax.set_xlabel(r"$v_1$"); ax.set_ylabel(r"$v_2$"); ax.set_zlabel(r"$t$")
    ax.view_init(elev=18, azim=-62)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ellipse_velocity_tube.png"), dpi=180)
    plt.close(fig)

    # -- Figure 2: selector inside the rotating ellipse (xi1, xi2, t) -------
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(projection="3d")
    for i, ti in enumerate(t_grid):
        if i % 6 == 0:
            rx, ry = ellipse_boundary(ti, 100)
            ax.plot(rx, ry, ti, color="tab:blue", lw=0.6, alpha=0.35)
    sc = ax.scatter(xi[:, 0], xi[:, 1], t_grid, c=level, cmap="viridis",
                    s=14, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label(r"level $\ell(t,\xi(t))$")
    ax.set_xlabel(r"$\xi_1$"); ax.set_ylabel(r"$\xi_2$"); ax.set_zlabel(r"$t$")
    ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ellipse_selector_level.png"), dpi=180)
    plt.close(fig)

    # -- Figure 3: state trajectory + level history -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(x[:, 0], x[:, 1], color="tab:blue", lw=2, label=r"$x(t)$")
    axes[0].scatter(*X0, color="tab:green", zorder=5, s=55, label=r"$x_0$")
    axes[0].scatter(*XF, color="tab:red", marker="*", zorder=5, s=140,
                    label=r"$x_{\rm f}$")
    axes[0].set_xlabel(r"$x_1$"); axes[0].set_ylabel(r"$x_2$")
    axes[0].set_title("State trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(t_grid, level, color="tab:blue", lw=2)
    axes[1].axhline(1.0, color="tab:red", linestyle="--", lw=1.2)
    axes[1].annotate(r"$\partial E(t)$: $\ell=1$", xy=(0.05, 1.0),
                     xytext=(0.05, 1.04), fontsize=11, color="tab:red")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$\ell(t,\xi(t))$")
    axes[1].set_title("Admissibility level of the selector")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ellipse_state_trajectory.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    apply_paper_style()
    z_opt, history = solve()
    xi, level = selector_from_z(z_opt)
    assert level.max() < 1.0, "admissibility-by-construction violated (bug)"
    make_figures(z_opt)
    print(f"\nFigures written to {os.path.normpath(OUT_DIR)}")
