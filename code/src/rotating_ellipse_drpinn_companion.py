#!/usr/bin/env python3
"""DR-PINN companion run for the rotating-ellipse steering problem (Sec. 6.2).

Answers the natural question "does the distance-residual approach itself work
on this benchmark?": trains a neural trajectory

    x_theta(t) = x0 + t * N_theta(t),        N_theta : R -> R^2  (hard IC)

with the genuine DR-PINN loss

    L(theta) = mean_i dist^2( xdot_theta(t_i) - g(t_i, x_theta(t_i)), E(t_i) )
               + lambda_f |x_theta(T) - x_f|^2,

where the projection onto the rotating ellipse E(t) has NO closed form: it is
computed pointwise by Newton iteration on the secular equation

    f(mu) = (a u / (a^2+mu))^2 + (b w / (b^2+mu))^2 - 1 = 0,   mu >= 0,

in the local frame (u,w) of the ellipse. Since f is convex and decreasing on
mu >= 0 with f(0) = level - 1 > 0 outside E(t), Newton from mu_0 = 0 converges
monotonically. Gradients flow through dist^2 = |v - Pi(v)|^2 with mu treated
as a constant (envelope theorem: grad_v dist^2 = 2 (v - Pi(v)), cf. the
gradient remark in the manuscript), implemented via tf.stop_gradient on mu.

Problem data (drift, ellipse, horizon, target) are identical to
rotating_ellipse_selector_experiment.py. Deterministic (seed 0). CPU, minutes.
"""

import os
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------------
# Problem data (identical to rotating_ellipse_selector_experiment.py)
# ----------------------------------------------------------------------------
T = 3.0
X0 = np.array([0.0, 0.0])
XF = np.array([1.8, 0.1])

LAMBDA_F = 10.0          # terminal steering weight in the DR-PINN loss
N_COL = 512              # collocation points (uniform grid on [0, T])
N_EPOCHS = 9000
LR = 2e-3
HIDDEN = 64
N_HIDDEN_LAYERS = 3
NEWTON_ITERS = 30
SEED = 0

DTYPE = tf.float64


def axes_ab_tf(t):
    a = 0.30 + 0.40 * tf.abs(tf.sin(np.pi * t / T))
    b = 0.15 + 0.25 * tf.abs(tf.cos(np.pi * t / T))
    return a, b


def angle_tf(t):
    return np.pi * t / T


def drift_tf(t, x):
    g1 = tf.sin(x[:, 0:1]) + 0.15 * tf.cos(2.0 * np.pi * t / T)
    g2 = 0.5 * tf.cos(x[:, 1:2]) - 0.20 * tf.sin(2.0 * np.pi * t / T)
    return tf.concat([g1, g2], axis=1)


def ellipse_dist2_tf(v, t):
    """dist^2(v, E(t)) elementwise; v: (N,2), t: (N,1). Differentiable in v.

    Newton on the secular equation with mu treated as constant in the
    backward pass (envelope theorem gives the exact gradient 2(v - Pi(v))).
    """
    a, b = axes_ab_tf(t)                       # (N,1)
    phi = angle_tf(t)
    c, s = tf.cos(phi), tf.sin(phi)
    # local frame coordinates (u, w) = R(-phi) v
    u = c * v[:, 0:1] + s * v[:, 1:2]
    w = -s * v[:, 0:1] + c * v[:, 1:2]

    level = (u / a) ** 2 + (w / b) ** 2
    outside = level > 1.0

    a2, b2 = a * a, b * b
    au2, bw2 = (a * u) ** 2, (b * w) ** 2
    mu = tf.zeros_like(u)
    for _ in range(NEWTON_ITERS):
        fa = au2 / (a2 + mu) ** 2 + bw2 / (b2 + mu) ** 2 - 1.0
        dfa = -2.0 * (au2 / (a2 + mu) ** 3 + bw2 / (b2 + mu) ** 3)
        mu = tf.where(outside, mu - fa / dfa, mu)
    mu = tf.stop_gradient(tf.maximum(mu, 0.0))

    up = a2 * u / (a2 + mu)
    wp = b2 * w / (b2 + mu)
    # rotate the projection back: Pi = R(phi) (up, wp)
    p1 = c * up - s * wp
    p2 = s * up + c * wp
    d2 = (v[:, 0:1] - p1) ** 2 + (v[:, 1:2] - p2) ** 2
    return tf.where(outside, d2, tf.zeros_like(d2)), level


def build_net():
    tf.keras.utils.set_random_seed(SEED)
    inp = tf.keras.Input(shape=(1,), dtype=DTYPE)
    h = inp
    for _ in range(N_HIDDEN_LAYERS):
        h = tf.keras.layers.Dense(HIDDEN, activation="tanh", dtype=DTYPE)(h)
    out = tf.keras.layers.Dense(2, dtype=DTYPE)(h)
    return tf.keras.Model(inp, out)


def x_theta(net, t):
    return tf.constant(X0, dtype=DTYPE)[None, :] + t * net(t)


def main():
    tf.keras.backend.set_floatx("float64")
    net = build_net()
    t_col = tf.constant(np.linspace(0.0, T, N_COL)[:, None], dtype=DTYPE)
    t_T = tf.constant([[T]], dtype=DTYPE)
    xf = tf.constant(XF, dtype=DTYPE)[None, :]

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LR, decay_steps=1000, decay_rate=0.80, staircase=True)
    # hold-then-decay: constant LR for the first 3000 epochs, then decay
    class HoldThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            return tf.where(step < 3000.0, LR, lr_schedule(step - 3000.0))
        def get_config(self):
            return {}
    opt = tf.keras.optimizers.Adam(learning_rate=HoldThenDecay())

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            with tf.GradientTape() as t1:
                t1.watch(t_col)
                x = x_theta(net, t_col)
            xdot = t1.batch_jacobian(x, t_col)[:, :, 0]
            v = xdot - drift_tf(t_col, x)
            d2, _ = ellipse_dist2_tf(v, t_col)
            term = tf.reduce_sum((x_theta(net, t_T) - xf) ** 2)
            loss = tf.reduce_mean(d2) + LAMBDA_F * term
        grads = tape.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(grads, net.trainable_variables))
        return loss, tf.reduce_mean(d2), term

    for ep in range(N_EPOCHS + 1):
        loss, incl, term = step()
        if ep % 1000 == 0:
            print(f"epoch {ep:5d}: loss={loss.numpy():.3e}  "
                  f"L_incl={incl.numpy():.3e}  |x(T)-x_f|^2={term.numpy():.3e}")

    # ---- evaluation on a dense grid -------------------------------------
    t_eval = tf.constant(np.linspace(0.0, T, 2000)[:, None], dtype=DTYPE)
    with tf.GradientTape() as t1:
        t1.watch(t_eval)
        x = x_theta(net, t_eval)
    xdot = t1.batch_jacobian(x, t_eval)[:, :, 0]
    v = (xdot - drift_tf(t_eval, x)).numpy()
    d2, level = ellipse_dist2_tf(tf.constant(v, dtype=DTYPE), t_eval)
    d = np.sqrt(np.maximum(d2.numpy(), 0.0))
    lev = level.numpy()
    xT = x_theta(net, t_T).numpy().flatten()
    err = float(np.linalg.norm(xT - XF))

    print("\n=== DR-PINN companion: final diagnostics (dense grid, 2000 pts) ===")
    print(f"terminal state x(T) = ({xT[0]:+.5f}, {xT[1]:+.5f}), "
          f"|x(T)-x_f| = {err:.3e}")
    print(f"inclusion residual dist(v, E(t)):  mean = {d.mean():.3e}   "
          f"max = {d.max():.3e}")
    print(f"fraction of grid points with v inside E(t): "
          f"{(lev <= 1.0).mean() * 100:.1f}%   "
          f"(mean level = {lev.mean():.3f}, max level = {lev.max():.3f})")


if __name__ == "__main__":
    main()
