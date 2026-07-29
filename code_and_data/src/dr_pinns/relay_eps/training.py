"""Training loop for one (lam, eps, seed, sampler) run.

Design notes:
  * identical optimiser protocol to the paper's baseline (Adam, hold-then-
    decay: 2e-3 held for 5000 epochs, then *0.85 every 3000 steps), so the
    eps-sweep isolates the effect of the multifunction, not of tuning;
  * the validation curve J_hat(epoch) -- mean distance residual on a fixed
    uniform batch -- is stored densely: it is the object that realises the
    hypothesis "J(u_n) -> 0" of Theorem 5.3 in the revised text;
  * pool history is stored for the hard-pool heat-map diagnostic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .model import RelayPINN
from .multifunction import dist2_relay_eps
from .sampling import CollocationSampler


class HoldThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, hold: int, decay_rate: float, decay_every: int):
        self.base_lr = base_lr
        self.hold = hold
        self.decay_rate = decay_rate
        self.decay_every = decay_every

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        decayed = self.base_lr * tf.pow(
            self.decay_rate, tf.math.floor((step - self.hold) / self.decay_every))
        return tf.where(step < self.hold, self.base_lr, decayed)

    def get_config(self):
        return dict(base_lr=self.base_lr, hold=self.hold,
                    decay_rate=self.decay_rate, decay_every=self.decay_every)


def train_run(cfg: dict, run_cfg: dict, outdir: Path) -> None:
    """cfg: global config dict; run_cfg: {lam, eps, seed, sampler}."""
    outdir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.set_floatx(cfg.get("dtype", "float32"))
    dtype = tf.keras.backend.floatx()

    lam = float(run_cfg["lam"])
    eps = float(run_cfg["eps"])
    seed = int(run_cfg["seed"])
    sampler_mode = run_cfg["sampler"]
    T = float(cfg["T"])

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = RelayPINN(width=cfg["width"], depth=cfg["depth"], seed=seed)

    sampler = CollocationSampler(
        T=T, mode=sampler_mode,
        batch_size=cfg["batch_size"],
        adaptive_frac=cfg["adaptive_frac"],
        pool_size=cfg["pool_size"],
        candidate_size=cfg["candidate_size"],
        refresh_every=cfg["refresh_every"],
        margin=cfg["pool_margin"],
        seed=seed + 1,
    )

    sched = HoldThenDecay(cfg["lr"], cfg["lr_hold"], cfg["lr_decay_rate"],
                          cfg["lr_decay_every"])
    opt = tf.keras.optimizers.Adam(learning_rate=sched)

    lam_c = tf.constant(lam, dtype=dtype)
    eps_c = tf.constant(eps, dtype=dtype)

    @tf.function
    def train_step(t, x, y):
        with tf.GradientTape() as tape:
            u, z = model.operator(t, x, y)
            r = dist2_relay_eps(u, z, lam_c, eps_c)
            loss = tf.reduce_mean(r)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_residual(t, x, y):
        u, z = model.operator(t, x, y)
        r = dist2_relay_eps(u, z, lam_c, eps_c)
        return tf.reduce_mean(r), tf.reduce_max(r)

    # fixed validation batch (uniform over Q), as in the paper
    rng = np.random.default_rng(10_000 + seed)
    vb = np.concatenate([rng.uniform(0, T, (cfg["val_size"], 1)),
                         rng.uniform(0, 1, (cfg["val_size"], 1)),
                         rng.uniform(0, 1, (cfg["val_size"], 1))], axis=1)
    vt = tf.convert_to_tensor(vb[:, :1], dtype); vx = tf.convert_to_tensor(vb[:, 1:2], dtype)
    vy = tf.convert_to_tensor(vb[:, 2:3], dtype)

    epochs = int(cfg["epochs"])
    hist_epochs, hist_train, hist_val_mean, hist_val_max = [], [], [], []
    t0 = time.time()

    for epoch in range(epochs):
        sampler.maybe_refresh(epoch, model, lam, eps, dist2_relay_eps)
        b = sampler.batch()
        loss = train_step(tf.convert_to_tensor(b[:, :1], dtype),
                          tf.convert_to_tensor(b[:, 1:2], dtype),
                          tf.convert_to_tensor(b[:, 2:3], dtype))
        if epoch % cfg["val_every"] == 0 or epoch == epochs - 1:
            vm, vx_max = eval_residual(vt, vx, vy)
            hist_epochs.append(epoch)
            hist_train.append(float(loss))
            hist_val_mean.append(float(vm))
            hist_val_max.append(float(vx_max))
            if epoch % (10 * cfg["val_every"]) == 0:
                print(f"[{outdir.name}] epoch {epoch:6d}  train {float(loss):.3e}  "
                      f"val_mean {float(vm):.3e}  val_max {float(vx_max):.3e}",
                      flush=True)

    wall = time.time() - t0
    model.save_weights_npz(outdir / "weights.npz")
    np.savez(outdir / "history.npz",
             epochs=np.array(hist_epochs), train=np.array(hist_train),
             val_mean=np.array(hist_val_mean), val_max=np.array(hist_val_max))
    if sampler.pool_history:
        np.savez(outdir / "pool_history.npz",
                 **{f"pool_{i}": p for i, p in enumerate(sampler.pool_history)})
    (outdir / "run_config.json").write_text(json.dumps(
        dict(run_cfg, wall_seconds=wall, epochs=epochs,
             final_val_mean=hist_val_mean[-1], final_val_max=hist_val_max[-1]),
        indent=2))
    print(f"[{outdir.name}] done in {wall/60:.1f} min; "
          f"final val_mean = {hist_val_mean[-1]:.3e}", flush=True)
