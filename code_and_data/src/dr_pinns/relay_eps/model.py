"""Hard-constrained trial solution and parabolic operator for Section 6.3(eps).

Trial solution (identical ansatz to the paper, eq. (6.26)):

    u_theta(t,x,y) = u0(x,y) + t * x(1-x) * y(1-y) * N_theta(t,x,y)

which enforces u_theta(0,.) = u0 and homogeneous Dirichlet data exactly.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def u0_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def u0_tf(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def build_network(width: int = 128, depth: int = 5, seed: int = 0) -> tf.keras.Model:
    """Fully connected tanh network N_theta : R^3 -> R, Glorot-normal init."""
    init = tf.keras.initializers.GlorotNormal(seed=seed)
    inp = tf.keras.Input(shape=(3,), dtype=tf.keras.backend.floatx())
    h = inp
    for _ in range(depth):
        h = tf.keras.layers.Dense(width, activation="tanh",
                                  kernel_initializer=init)(h)
    out = tf.keras.layers.Dense(1, kernel_initializer=init)(h)
    return tf.keras.Model(inp, out)


class RelayPINN:
    """Wraps the network with the hard-constraint ansatz and the operator."""

    def __init__(self, width: int = 128, depth: int = 5, seed: int = 0):
        self.net = build_network(width=width, depth=depth, seed=seed)

    @property
    def trainable_variables(self):
        return self.net.trainable_variables

    def u(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """u_theta(t,x,y); all inputs shape (N,1)."""
        n = self.net(tf.concat([t, x, y], axis=1))
        bubble = x * (1.0 - x) * y * (1.0 - y)
        return u0_tf(x, y) + t * bubble * n

    def operator(self, t: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
        """Returns (u, z) with z = d_t u - Laplace u, via nested tapes."""
        with tf.GradientTape(persistent=True) as t2:
            t2.watch([x, y])
            with tf.GradientTape(persistent=True) as t1:
                t1.watch([t, x, y])
                u = self.u(t, x, y)
            u_t = t1.gradient(u, t)
            u_x = t1.gradient(u, x)
            u_y = t1.gradient(u, y)
        u_xx = t2.gradient(u_x, x)
        u_yy = t2.gradient(u_y, y)
        del t1, t2
        z = u_t - (u_xx + u_yy)
        return u, z

    # ---------- weight (de)serialisation, TF-version-robust ----------

    def save_weights_npz(self, path: str) -> None:
        arrs = [v.numpy() for v in self.net.weights]
        np.savez(path, *arrs)

    def load_weights_npz(self, path: str) -> None:
        data = np.load(path)
        arrs = [data[k] for k in sorted(data.files, key=lambda s: int(s.split("_")[1]))]
        for v, a in zip(self.net.weights, arrs):
            v.assign(tf.cast(a, v.dtype))  # v.dtype is a str under Keras 3
