"""Dead-zone (boundary-layer) relay multifunction Phi_eps and its distance residual.

    Phi_eps(s) = {-lam}        if s >  eps,
                 [-lam, lam]   if |s| <= eps,
                 {+lam}        if s < -eps.

For eps = 0 this reduces to the exact relay multifunction of the original
Section 6.3 (the interval branch then has probability zero for a smooth,
generic network output, exactly as in the paper).

Properties used by the paper:
  * (P1)-(P3) of Section 5.2 hold verbatim (m0 = lam), so Theorem 5.3 applies.
  * f_eps(s) = -lam * sat(s/eps) is a continuous selection, so existence also
    follows directly from Theorem 2.12 (no maximal-monotone detour needed).
  * Phi(s) [subset] Phi_eps(s) pointwise, hence the eps=0 relay reference
    solution remains an admissible element of S_eps(u0).

Gradient convention (matches Remark 4.2 / the "hold the projection constant"
practice used throughout the paper): within each branch the squared distance
is differentiated exactly w.r.t. the operator value z; the dependence of the
branch selection on the state s is piecewise constant, so its a.e. derivative
is zero and `tf.where` propagates exactly the within-branch gradient.
"""

from __future__ import annotations

import tensorflow as tf


def dist2_relay_eps(s: tf.Tensor, z: tf.Tensor, lam: float, eps: float) -> tf.Tensor:
    """Pointwise squared distance dist^2(z, Phi_eps(s)).

    Branches:
      s >  eps : (z + lam)^2
      s < -eps : (z - lam)^2
      |s|<= eps: relu(|z| - lam)^2
    """
    lam = tf.cast(lam, s.dtype)
    eps = tf.cast(eps, s.dtype)
    d_plus = tf.square(z + lam)
    d_minus = tf.square(z - lam)
    d_band = tf.square(tf.nn.relu(tf.abs(z) - lam))
    return tf.where(s > eps, d_plus, tf.where(s < -eps, d_minus, d_band))


def project_relay_eps(s: tf.Tensor, z: tf.Tensor, lam: float, eps: float) -> tf.Tensor:
    """Metric projection of z onto Phi_eps(s) (diagnostics only)."""
    lam = tf.cast(lam, s.dtype)
    eps = tf.cast(eps, s.dtype)
    w_plus = -lam * tf.ones_like(z)
    w_minus = lam * tf.ones_like(z)
    w_band = tf.clip_by_value(z, -lam, lam)
    return tf.where(s > eps, w_plus, tf.where(s < -eps, w_minus, w_band))


def branch_id(s: tf.Tensor, eps: float) -> tf.Tensor:
    """Integer branch label: +1 (s>eps), 0 (band), -1 (s<-eps)."""
    eps = tf.cast(eps, s.dtype)
    return tf.where(
        s > eps,
        tf.ones_like(s, dtype=tf.int32),
        tf.where(s < -eps,
                 -tf.ones_like(s, dtype=tf.int32),
                 tf.zeros_like(s, dtype=tf.int32)),
    )
