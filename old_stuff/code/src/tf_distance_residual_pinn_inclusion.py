"""
TensorFlow/Keras implementation of a distance-residual PINN
for a scalar differential inclusion.

Problem:
    x'(t) in F(x(t)),   x(0)=0,   t in [0,1],

where
    F(x) = [-x - 1, -x + 1].

We impose the terminal condition
    x(1) = 0.5.

A known feasible solution is
    x_*(t) = u_*(1 - exp(-t)),
    u_* = 0.5 / (1 - exp(-1)).

The PINN minimizes the squared distance residual

    dist^2(x_theta'(t), F(x_theta(t))).

For F(x) = [a,b], with
    a = -x - 1,
    b = -x + 1,

the squared distance is

    ReLU(a - x')^2 + ReLU(x' - b)^2.
"""

import os
import math
import argparse
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


@dataclass
class Config:
    seed: int = 1234
    n_collocation: int = 256
    n_test: int = 1000
    width: int = 32
    depth: int = 3
    lr: float = 1e-3
    epochs: int = 12000
    lambda_terminal: float = 100.0
    terminal_value: float = 0.5
    output_dir: str = "results_tf_distance_residual_pinn"


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_base_network(width: int = 32, depth: int = 3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(1,))
    z = inputs

    for _ in range(depth):
        z = tf.keras.layers.Dense(
            width,
            activation="tanh",
            kernel_initializer="glorot_normal",
            bias_initializer="zeros",
        )(z)

    outputs = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer="glorot_normal",
        bias_initializer="zeros",
    )(z)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


class TrialSolution(tf.keras.Model):
    """
    Hard imposition of x(0)=0 through

        x_theta(t) = t N_theta(t).
    """

    def __init__(self, width: int = 32, depth: int = 3):
        super().__init__()
        self.N = build_base_network(width=width, depth=depth)

    def call(self, t, training=False):
        return t * self.N(t, training=training)


def model_and_derivative(model: tf.keras.Model, t: tf.Tensor, training: bool = False):
    """
    Computes x_theta(t) and x_theta'(t) with automatic differentiation.
    """
    with tf.GradientTape() as tape:
        tape.watch(t)
        x = model(t, training=training)
    dx = tape.gradient(x, t)
    return x, dx


def distance_residual(model: tf.keras.Model, t: tf.Tensor, training: bool = False):
    """
    Computes the pointwise squared distance residual:

        dist^2(x_theta'(t), [-x_theta(t)-1, -x_theta(t)+1]).
    """
    x, dx = model_and_derivative(model, t, training=training)

    lower = -x - 1.0
    upper = -x + 1.0

    residual_sq = tf.nn.relu(lower - dx) ** 2 + tf.nn.relu(dx - upper) ** 2

    # Since x' = -x + u, the implicit control is u = x' + x.
    u_reconstructed = dx + x

    return residual_sq, x, dx, lower, upper, u_reconstructed


def compute_loss(model: tf.keras.Model, t_col: tf.Tensor, cfg: Config):
    residual_sq, _, _, _, _, _ = distance_residual(model, t_col, training=True)

    t_terminal = tf.ones((1, 1), dtype=tf.float32)
    x_terminal = model(t_terminal, training=True)

    loss_inclusion = tf.reduce_mean(residual_sq)
    loss_terminal = tf.reduce_mean((x_terminal - cfg.terminal_value) ** 2)

    total_loss = loss_inclusion + cfg.lambda_terminal * loss_terminal
    return total_loss, loss_inclusion, loss_terminal


@tf.function
def train_step(model, optimizer, t_col, lambda_terminal, terminal_value):
    """
    A compiled training step. The scalar parameters are passed explicitly
    so that TensorFlow does not need to trace over the dataclass.
    """
    with tf.GradientTape() as tape:
        residual_sq, _, _, _, _, _ = distance_residual(model, t_col, training=True)

        t_terminal = tf.ones((1, 1), dtype=tf.float32)
        x_terminal = model(t_terminal, training=True)

        loss_inclusion = tf.reduce_mean(residual_sq)
        loss_terminal = tf.reduce_mean((x_terminal - terminal_value) ** 2)
        total_loss = loss_inclusion + lambda_terminal * loss_terminal

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, loss_inclusion, loss_terminal


def exact_solution(t: np.ndarray, terminal_value: float = 0.5):
    u_star = terminal_value / (1.0 - math.exp(-1.0))
    x_star = u_star * (1.0 - np.exp(-t))
    return x_star, u_star


def train(model: tf.keras.Model, cfg: Config):
    t_col = tf.linspace(0.0, 1.0, cfg.n_collocation)
    t_col = tf.reshape(t_col, (-1, 1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

    history = {
        "loss_total": [],
        "loss_inclusion": [],
        "loss_terminal": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        loss, loss_inclusion, loss_terminal = train_step(
            model,
            optimizer,
            t_col,
            tf.constant(cfg.lambda_terminal, dtype=tf.float32),
            tf.constant(cfg.terminal_value, dtype=tf.float32),
        )

        history["loss_total"].append(float(loss.numpy()))
        history["loss_inclusion"].append(float(loss_inclusion.numpy()))
        history["loss_terminal"].append(float(loss_terminal.numpy()))

        if epoch == 1 or epoch % 1000 == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"loss={history['loss_total'][-1]:.3e} | "
                f"incl={history['loss_inclusion'][-1]:.3e} | "
                f"terminal={history['loss_terminal'][-1]:.3e}"
            )

    return history


def evaluate(model: tf.keras.Model, cfg: Config):
    t = tf.linspace(0.0, 1.0, cfg.n_test)
    t = tf.reshape(t, (-1, 1))

    residual_sq, x, dx, lower, upper, u = distance_residual(model, t, training=False)

    t_np = t.numpy().reshape(-1)
    x_np = x.numpy().reshape(-1)
    dx_np = dx.numpy().reshape(-1)
    lower_np = lower.numpy().reshape(-1)
    upper_np = upper.numpy().reshape(-1)
    u_np = u.numpy().reshape(-1)
    residual_np = residual_sq.numpy().reshape(-1)

    x_exact, u_star = exact_solution(t_np, cfg.terminal_value)

    metrics = {
        "mean_distance_residual": float(np.mean(residual_np)),
        "max_distance_residual": float(np.max(residual_np)),
        "terminal_error": float(abs(x_np[-1] - cfg.terminal_value)),
        "l2_error_exact": float(np.sqrt(np.mean((x_np - x_exact) ** 2))),
        "u_star_exact": float(u_star),
        "u_reconstructed_min": float(np.min(u_np)),
        "u_reconstructed_max": float(np.max(u_np)),
    }

    data = {
        "t": t_np,
        "x": x_np,
        "dx": dx_np,
        "lower": lower_np,
        "upper": upper_np,
        "u": u_np,
        "residual_sq": residual_np,
        "x_exact": x_exact,
        "u_star": u_star,
    }

    return data, metrics


def save_plots(data, history, cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    t = data["t"]

    plt.figure(figsize=(7, 4))
    plt.plot(t, data["x"], label=r"$x_\theta(t)$")
    plt.plot(t, data["x_exact"], "--", label=r"$x_*(t)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.title("State trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "state_vs_exact.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, data["dx"], label=r"$\dot{x}_\theta(t)$")
    plt.plot(t, data["lower"], "--", label=r"$-x_\theta(t)-1$")
    plt.plot(t, data["upper"], "--", label=r"$-x_\theta(t)+1$")
    plt.xlabel(r"$t$")
    plt.ylabel("velocity")
    plt.title("Derivative and admissible interval")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "derivative_admissible_band.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(t, data["residual_sq"] + 1e-20)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\operatorname{dist}^2(\dot{x}_\theta(t),F(x_\theta(t)))$")
    plt.title("Pointwise squared distance residual")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "pointwise_residual.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, data["u"], label=r"$u_\theta(t)=\dot{x}_\theta(t)+x_\theta(t)$")
    plt.axhline(1.0, linestyle="--", label=r"$u=1$")
    plt.axhline(-1.0, linestyle="--", label=r"$u=-1$")
    plt.axhline(data["u_star"], linestyle=":", label=rf"$u_*={data['u_star']:.3f}$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u(t)$")
    plt.title("Reconstructed implicit control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "reconstructed_control.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(history["loss_total"], label="total")
    plt.semilogy(history["loss_inclusion"], label="inclusion")
    plt.semilogy(history["loss_terminal"], label="terminal")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "training_history.png"), dpi=200)
    plt.close()


def save_metrics(metrics, cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, "metrics.txt")

    with open(path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.12e}\n")

    print("\nMetrics")
    print("-------")
    for key, value in metrics.items():
        print(f"{key}: {value:.12e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorFlow distance-residual PINN for a scalar differential inclusion."
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-collocation", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--lambda-terminal", type=float, default=100.0)
    parser.add_argument("--terminal-value", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="results_tf_distance_residual_pinn")

    args = parser.parse_args()

    return Config(
        seed=args.seed,
        n_collocation=args.n_collocation,
        n_test=args.n_test,
        width=args.width,
        depth=args.depth,
        lr=args.lr,
        epochs=args.epochs,
        lambda_terminal=args.lambda_terminal,
        terminal_value=args.terminal_value,
        output_dir=args.output_dir,
    )


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Configuration")
    print("-------------")
    print(cfg)

    model = TrialSolution(width=cfg.width, depth=cfg.depth)

    history = train(model, cfg)
    data, metrics = evaluate(model, cfg)

    save_plots(data, history, cfg)
    save_metrics(metrics, cfg)

    model.save_weights(os.path.join(cfg.output_dir, "model.weights.h5"))

    print(f"\nSaved results in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
