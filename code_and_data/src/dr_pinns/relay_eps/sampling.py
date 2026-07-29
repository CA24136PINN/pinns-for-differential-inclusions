"""Collocation samplers for the parabolic relay benchmark.

Three modes (the ablation requested by the referees):

  * "uniform" : plain uniform collocation over Q = (0,T) x (0,1)^2.
  * "front"   : the paper's state-based adaptive scheme (smallest |u_theta|),
                *with the boundary-layer confounder removed*: candidate points
                closer than `margin` to the spatial boundary are excluded from
                the hard pool, because the ansatz forces u_theta = 0 there
                regardless of the extinction front.
  * "rar"     : residual-based RAR in the sense of Lu et al. (largest
                pointwise distance residual), same margin exclusion for a
                fair comparison.

The hard pool is refreshed every `refresh_every` epochs; a fraction
`adaptive_frac` of every batch is drawn from the pool, the rest uniformly
over Q to preserve global coverage. All pools/batches are also returned to
the caller for the hard-pool heat-map diagnostic figure.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class CollocationSampler:
    def __init__(self, T: float, mode: str, batch_size: int,
                 adaptive_frac: float, pool_size: int, candidate_size: int,
                 refresh_every: int, margin: float, seed: int):
        assert mode in ("uniform", "front", "rar")
        self.T = float(T)
        self.mode = mode
        self.batch_size = int(batch_size)
        self.n_adapt = int(round(adaptive_frac * batch_size)) if mode != "uniform" else 0
        self.n_unif = self.batch_size - self.n_adapt
        self.pool_size = int(pool_size)
        self.candidate_size = int(candidate_size)
        self.refresh_every = int(refresh_every)
        self.margin = float(margin)
        self.rng = np.random.default_rng(seed)
        self.pool: np.ndarray | None = None  # (pool_size, 3) columns t,x,y
        self.pool_history: list[np.ndarray] = []

    # ------------------------------------------------------------------

    def _uniform(self, n: int) -> np.ndarray:
        t = self.rng.uniform(0.0, self.T, size=(n, 1))
        x = self.rng.uniform(0.0, 1.0, size=(n, 1))
        y = self.rng.uniform(0.0, 1.0, size=(n, 1))
        return np.concatenate([t, x, y], axis=1)

    def _interior_mask(self, pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 1], pts[:, 2]
        d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        return d >= self.margin

    def refresh_pool(self, model, lam: float, eps: float,
                     dist2_fn, eval_batch: int = 4096) -> None:
        """Rebuild the hard pool from a fresh uniform candidate draw."""
        if self.mode == "uniform":
            return
        cand = self._uniform(self.candidate_size)
        cand = cand[self._interior_mask(cand)]
        scores = np.empty(len(cand))
        dtype = tf.keras.backend.floatx()
        for i in range(0, len(cand), eval_batch):
            b = cand[i:i + eval_batch]
            t = tf.convert_to_tensor(b[:, :1], dtype=dtype)
            x = tf.convert_to_tensor(b[:, 1:2], dtype=dtype)
            y = tf.convert_to_tensor(b[:, 2:3], dtype=dtype)
            if self.mode == "front":
                u = model.u(t, x, y)
                scores[i:i + eval_batch] = np.abs(u.numpy()[:, 0])
            else:  # rar
                u, z = model.operator(t, x, y)
                r = dist2_fn(u, z, lam, eps)
                scores[i:i + eval_batch] = -r.numpy()[:, 0]  # negate: keep largest
        k = min(self.pool_size, len(cand))
        idx = np.argpartition(scores, k - 1)[:k]
        self.pool = cand[idx]
        self.pool_history.append(self.pool.copy())

    def batch(self) -> np.ndarray:
        pts = self._uniform(self.n_unif if self.pool is not None else self.batch_size)
        if self.pool is not None and self.n_adapt > 0:
            j = self.rng.integers(0, len(self.pool), size=self.n_adapt)
            pts = np.concatenate([pts, self.pool[j]], axis=0)
        return pts

    def maybe_refresh(self, epoch: int, model, lam: float, eps: float, dist2_fn) -> None:
        if self.mode == "uniform":
            return
        if epoch % self.refresh_every == 0:
            self.refresh_pool(model, lam, eps, dist2_fn)
