#!/usr/bin/env python3
"""Correct DR-PINN companion for the rotating-ellipse steering problem.

This experiment solves the two-point differential inclusion

    xdot(t) in g(t, x(t)) + E(t),
    x(0) = x0,  x(T) = xf,

with a Distance-Residual PINN.  Both endpoint constraints are enforced by

    x_theta(t) = x0 + (t/T)(xf-x0) + t(T-t) N_theta(t),

and the only training term is

    mean_i dist^2(xdot_theta(t_i)-g(t_i,x_theta(t_i)), E(t_i)).

The metric projection onto E(t) is found by Newton iteration on the ellipse
secular equation.  The *whole projection point* is detached in the backward
pass.  This is essential: detaching only the Lagrange multiplier is not the
envelope gradient.  With the projection point treated as constant, autograd
returns exactly 2(v-Pi_E(v)) with respect to v.

Outputs are written to code/results/results_rotating_ellipse/:
  - drpinn_companion_seed<seed>_history.csv
  - drpinn_companion_seed<seed>_metrics.json
  - drpinn_companion_seed<seed>.png
  - drpinn_companion_seed<seed>.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply_paper_style

T = 3.0
X0 = np.array([0.0, 0.0], dtype=np.float64)
XF = np.array([1.8, 0.1], dtype=np.float64)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    n_collocation: int = 512
    n_eval: int = 2000
    epochs: int = 7000
    learning_rate: float = 2e-3
    hidden_width: int = 64
    hidden_layers: int = 3
    newton_iterations: int = 30
    log_every: int = 1000
    dtype: str = "float64"


class MLPWithDerivative(torch.nn.Module):
    """Tanh MLP with an analytic derivative with respect to scalar time."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.hidden = torch.nn.ModuleList()
        in_dim = 1
        for _ in range(depth):
            layer = torch.nn.Linear(in_dim, width)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            in_dim = width
        self.output = torch.nn.Linear(in_dim, 2)
        torch.nn.init.xavier_normal_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward_with_derivative(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = t
        dh = torch.ones_like(t)
        for layer in self.hidden:
            z = layer(h)
            dz = dh @ layer.weight.T
            h = torch.tanh(z)
            dh = (1.0 - h.square()) * dz
        out = self.output(h)
        dout = dh @ self.output.weight.T
        return out, dout

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward_with_derivative(t)[0]


def axes_ab(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = 0.30 + 0.40 * torch.abs(torch.sin(math.pi * t / T))
    b = 0.15 + 0.25 * torch.abs(torch.cos(math.pi * t / T))
    return a, b


def drift(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g1 = torch.sin(x[:, 0:1]) + 0.15 * torch.cos(2.0 * math.pi * t / T)
    g2 = 0.5 * torch.cos(x[:, 1:2]) - 0.20 * torch.sin(2.0 * math.pi * t / T)
    return torch.cat([g1, g2], dim=1)


def trajectory_and_velocity(
    network: MLPWithDerivative,
    t: torch.Tensor,
    x0: torch.Tensor,
    xf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Endpoint-hard trial trajectory and its exact time derivative."""
    n, dn_dt = network.forward_with_derivative(t)
    tau = t / T
    envelope = t * (T - t)
    x = x0[None, :] + tau * (xf - x0)[None, :] + envelope * n
    xdot = (xf - x0)[None, :] / T + (T - 2.0 * t) * n + envelope * dn_dt
    return x, xdot


def detached_ellipse_projection(
    v: torch.Tensor,
    t: torch.Tensor,
    newton_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Metric projection onto E(t), detached as required by the envelope theorem.

    Returns (projection, level(v), outside-mask).  The projection computation is
    deliberately outside the autograd graph.  The loss is then formed from the
    original v and this detached projection.
    """
    with torch.no_grad():
        vd = v.detach()
        td = t.detach()
        a, b = axes_ab(td)
        phi = math.pi * td / T
        c, s = torch.cos(phi), torch.sin(phi)

        u = c * vd[:, 0:1] + s * vd[:, 1:2]
        w = -s * vd[:, 0:1] + c * vd[:, 1:2]
        level = (u / a).square() + (w / b).square()
        outside = level > 1.0 + 1e-12

        a2, b2 = a.square(), b.square()
        au2, bw2 = (a * u).square(), (b * w).square()
        mu = torch.zeros_like(u)
        for _ in range(newton_iterations):
            f = au2 / (a2 + mu).square() + bw2 / (b2 + mu).square() - 1.0
            df = -2.0 * (
                au2 / (a2 + mu).pow(3) + bw2 / (b2 + mu).pow(3)
            )
            safe_df = torch.where(
                torch.abs(df) > 1e-30,
                df,
                torch.full_like(df, -1e-30),
            )
            mu = torch.where(outside, mu - f / safe_df, mu)
        mu = torch.clamp(mu, min=0.0)

        up = a2 * u / (a2 + mu)
        wp = b2 * w / (b2 + mu)
        projection = torch.cat([c * up - s * wp, s * up + c * wp], dim=1)

    return projection, level, outside


def distance_squared(
    v: torch.Tensor,
    t: torch.Tensor,
    newton_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection, level, outside = detached_ellipse_projection(v, t, newton_iterations)
    d2 = (v - projection).square().sum(dim=1, keepdim=True)
    return torch.where(outside, d2, torch.zeros_like(d2)), level


def learning_rate(epoch: int, initial: float) -> float:
    if epoch < 3000:
        return initial
    return initial * 0.80 ** ((epoch - 3000) // 1000)


def gradient_check(dtype: torch.dtype, newton_iterations: int) -> dict[str, float]:
    """Finite-difference check of the projection/envelope gradient."""
    t = torch.tensor([[1.1]], dtype=dtype)
    v = torch.tensor([[0.9, -0.4]], dtype=dtype, requires_grad=True)
    d2, _ = distance_squared(v, t, newton_iterations)
    d2.sum().backward()
    analytic = v.grad.detach().cpu().numpy().reshape(-1)

    h = 1e-6 if dtype == torch.float64 else 2e-4
    fd = []
    for j in range(2):
        direction = torch.zeros_like(v)
        direction[0, j] = h
        plus, _ = distance_squared(v.detach() + direction, t, newton_iterations)
        minus, _ = distance_squared(v.detach() - direction, t, newton_iterations)
        fd.append(float((plus - minus).item() / (2.0 * h)))
    fd_arr = np.asarray(fd)
    rel = float(np.linalg.norm(analytic - fd_arr) / max(np.linalg.norm(fd_arr), 1e-15))
    if rel > 2e-5:
        raise RuntimeError(f"ellipse distance gradient check failed: relative error {rel:.3e}")
    return {
        "gradient_check_relative_error": rel,
        "gradient_check_autograd_0": float(analytic[0]),
        "gradient_check_autograd_1": float(analytic[1]),
        "gradient_check_fd_0": float(fd_arr[0]),
        "gradient_check_fd_1": float(fd_arr[1]),
    }


def train(config: Config, out_dir: str) -> dict[str, float | int | str]:
    if config.dtype != "float64":
        raise ValueError("The paper experiment is defined in float64.")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.set_flush_denormal(True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    x0 = torch.tensor(X0, dtype=dtype)
    xf = torch.tensor(XF, dtype=dtype)
    network = MLPWithDerivative(config.hidden_width, config.hidden_layers)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    t_col = torch.linspace(0.0, T, config.n_collocation, dtype=dtype).reshape(-1, 1)

    check = gradient_check(dtype, config.newton_iterations)
    history: list[tuple[int, float, float]] = []
    start = time.perf_counter()

    for epoch in range(config.epochs + 1):
        lr = learning_rate(epoch, config.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        x, xdot = trajectory_and_velocity(network, t_col, x0, xf)
        v = xdot - drift(t_col, x)
        d2, _ = distance_squared(v, t_col, config.newton_iterations)
        loss = d2.mean()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append((epoch, loss_value, lr))
        if epoch % config.log_every == 0:
            print(f"epoch {epoch:5d}: inclusion loss = {loss_value:.3e}  lr = {lr:.3e}")

    elapsed = time.perf_counter() - start

    t_eval = torch.linspace(0.0, T, config.n_eval, dtype=dtype).reshape(-1, 1)
    x_eval, xdot_eval = trajectory_and_velocity(network, t_eval, x0, xf)
    v_eval = xdot_eval - drift(t_eval, x_eval)
    d2_eval, level_eval = distance_squared(v_eval, t_eval, config.newton_iterations)
    distance_eval = torch.sqrt(torch.clamp(d2_eval, min=0.0)).detach().cpu().numpy().reshape(-1)
    level_np = level_eval.detach().cpu().numpy().reshape(-1)
    x_np = x_eval.detach().cpu().numpy()
    v_np = v_eval.detach().cpu().numpy()

    x_start = x_np[0]
    x_final = x_np[-1]
    metrics: dict[str, float | int | str] = {
        **asdict(config),
        **check,
        "training_seconds": float(elapsed),
        "final_collocation_loss": float(history[-1][1]),
        "initial_endpoint_error": float(np.linalg.norm(x_start - X0)),
        "terminal_endpoint_error": float(np.linalg.norm(x_final - XF)),
        "distance_mean_dense": float(distance_eval.mean()),
        "distance_rms_dense": float(np.sqrt(np.mean(distance_eval**2))),
        "distance_max_dense": float(distance_eval.max()),
        "inside_fraction_dense": float(np.mean(level_np <= 1.0)),
        "level_mean_dense": float(level_np.mean()),
        "level_max_dense": float(level_np.max()),
    }

    os.makedirs(out_dir, exist_ok=True)
    stem = f"drpinn_companion_seed{config.seed}"
    history_path = os.path.join(out_dir, f"{stem}_history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "inclusion_loss", "learning_rate"])
        writer.writerows(history)

    metrics_path = os.path.join(out_dir, f"{stem}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    torch.save(
        {"config": asdict(config), "state_dict": network.state_dict()},
        os.path.join(out_dir, f"{stem}.pt"),
    )

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(x_np[:, 0], x_np[:, 1], lw=2.0, label=r"$x_\theta(t)$")
    axes[0].scatter(*X0, s=55, zorder=5, label=r"$x_0$")
    axes[0].scatter(*XF, marker="*", s=140, zorder=5, label=r"$x_{\rm f}$")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].set_title("Endpoint-hard DR-PINN trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(t_eval.cpu().numpy().reshape(-1), np.maximum(distance_eval, 1e-16), lw=1.8)
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$\operatorname{dist}(\dot x_\theta-g,E(t))$")
    axes[1].set_title("Dense-grid inclusion residual")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    figure_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("\n=== Correct DR-PINN companion diagnostics ===")
    for key in (
        "final_collocation_loss",
        "initial_endpoint_error",
        "terminal_endpoint_error",
        "distance_mean_dense",
        "distance_rms_dense",
        "distance_max_dense",
        "inside_fraction_dense",
        "level_mean_dense",
        "level_max_dense",
        "gradient_check_relative_error",
        "training_seconds",
    ):
        print(f"{key}: {metrics[key]}")
    print(f"outputs: {os.path.normpath(out_dir)}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--n-collocation", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "ELLIPSE_OUT_DIR",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "results",
                "results_rotating_ellipse",
            ),
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        n_collocation=args.n_collocation,
        n_eval=args.n_eval,
    )
    train(config, args.out_dir)


if __name__ == "__main__":
    main()
