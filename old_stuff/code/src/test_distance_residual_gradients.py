#!/usr/bin/env python3
"""Regression checks for the two gradient bugs found in the ODE experiments.

1. A state-dependent translated set F(x)=Ax+U must retain the derivative of Ax.
2. For an ellipse projection computed numerically, the whole projected point -
   not only the Lagrange multiplier - must be detached to obtain the envelope
   gradient 2(v-Pi(v)).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotating_ellipse_drpinn_companion import distance_squared, gradient_check


def qp_project(q: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    k = len(vertices)
    gram = vertices @ vertices.T
    linear = vertices @ q
    result = minimize(
        lambda lam: float(lam @ gram @ lam - 2.0 * linear @ lam),
        np.ones(k) / k,
        jac=lambda lam: 2.0 * (gram @ lam - linear),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda lam: lam.sum() - 1.0},
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    lam = np.clip(result.x, 0.0, 1.0)
    lam /= lam.sum()
    return vertices.T @ lam


def translated_set_gradient_check() -> dict[str, np.ndarray | float]:
    a = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    vertices = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    x = np.array([0.7, -0.2])
    xdot = np.array([2.0, 0.5])
    q = xdot - a @ x
    projection = qp_project(q, vertices)
    residual = q - projection
    analytic_x = -2.0 * a.T @ residual
    analytic_xdot = 2.0 * residual

    h = 1e-6

    def value(x_value: np.ndarray, xdot_value: np.ndarray) -> float:
        q_value = xdot_value - a @ x_value
        p_value = qp_project(q_value, vertices)
        return float(np.sum((q_value - p_value) ** 2))

    fd_x = np.array(
        [
            (
                value(x + h * np.eye(2)[j], xdot)
                - value(x - h * np.eye(2)[j], xdot)
            )
            / (2.0 * h)
            for j in range(2)
        ]
    )
    fd_xdot = np.array(
        [
            (
                value(x, xdot + h * np.eye(2)[j])
                - value(x, xdot - h * np.eye(2)[j])
            )
            / (2.0 * h)
            for j in range(2)
        ]
    )
    rel_x = float(np.linalg.norm(analytic_x - fd_x) / np.linalg.norm(fd_x))
    rel_xdot = float(
        np.linalg.norm(analytic_xdot - fd_xdot) / np.linalg.norm(fd_xdot)
    )
    if rel_x > 1e-6 or rel_xdot > 1e-6:
        raise AssertionError((rel_x, rel_xdot))
    return {
        "analytic_x": analytic_x,
        "finite_difference_x": fd_x,
        "analytic_xdot": analytic_xdot,
        "finite_difference_xdot": fd_xdot,
        "relative_error_x": rel_x,
        "relative_error_xdot": rel_xdot,
    }


def wrong_ellipse_gradient() -> tuple[np.ndarray, np.ndarray]:
    """Show that detaching mu alone does not match finite differences."""
    torch.set_default_dtype(torch.float64)
    t = torch.tensor([[1.1]])
    v = torch.tensor([[0.9, -0.4]], requires_grad=True)

    # Recreate the old backward path: mu detached, projection not detached.
    a = 0.30 + 0.40 * torch.abs(torch.sin(math.pi * t / 3.0))
    b = 0.15 + 0.25 * torch.abs(torch.cos(math.pi * t / 3.0))
    phi = math.pi * t / 3.0
    c, s = torch.cos(phi), torch.sin(phi)
    u = c * v[:, 0:1] + s * v[:, 1:2]
    w = -s * v[:, 0:1] + c * v[:, 1:2]
    a2, b2 = a.square(), b.square()
    mu = torch.zeros_like(u)
    for _ in range(40):
        f = (a * u).square() / (a2 + mu).square() + (b * w).square() / (
            b2 + mu
        ).square() - 1.0
        df = -2.0 * (
            (a * u).square() / (a2 + mu).pow(3)
            + (b * w).square() / (b2 + mu).pow(3)
        )
        mu = mu - f / df
    mu = mu.detach()
    up = a2 * u / (a2 + mu)
    wp = b2 * w / (b2 + mu)
    projection = torch.cat([c * up - s * wp, s * up + c * wp], dim=1)
    old_loss = (v - projection).square().sum()
    old_loss.backward()
    old_gradient = v.grad.detach().numpy().reshape(-1)

    v2 = torch.tensor([[0.9, -0.4]], requires_grad=True)
    correct_loss, _ = distance_squared(v2, t, 40)
    correct_loss.sum().backward()
    correct_gradient = v2.grad.detach().numpy().reshape(-1)
    return old_gradient, correct_gradient


def main() -> None:
    translated = translated_set_gradient_check()
    ellipse = gradient_check(torch.float64, 40)
    old_gradient, correct_gradient = wrong_ellipse_gradient()
    mismatch = float(
        np.linalg.norm(old_gradient - correct_gradient)
        / np.linalg.norm(correct_gradient)
    )
    if mismatch < 1e-2:
        raise AssertionError("old ellipse gradient unexpectedly matches the correct one")

    print("translated-set gradient check:")
    for key, value in translated.items():
        print(f"  {key}: {value}")
    print("ellipse envelope-gradient check:")
    for key, value in ellipse.items():
        print(f"  {key}: {value}")
    print(f"  old stop-mu-only gradient: {old_gradient}")
    print(f"  correct detached-projection gradient: {correct_gradient}")
    print(f"  relative mismatch of old gradient: {mismatch:.6f}")


if __name__ == "__main__":
    main()
