"""Post-training evaluation of a single run.

Produces per-run JSON with:
  * dense-grid residual: mean / max over M uniform points, plus the
    referee-requested stratification pre-front / front / post-front
    (time strata relative to the reference extinction time t*_ref);
  * band-entry times: t_band(eps) = first grid time with sup_x |u_theta| <= eps,
    and its persistent variant (stays inside the band until T) -- the
    theory-native replacement of the arbitrary 1e-3 threshold;
  * L2 / Linf norm curves on an evaluation grid, and the pre-extinction
    discrepancy vs the relay reference;
  * branch-selection diagnostic: (s, z) cloud, violation count
    #{dist2 > tol} / N (binary check against the graph of Phi_eps);
  * non-uniqueness witness distances: || u_theta(T) ||, || u_theta(T) - lam*v1 ||
    (the latter meaningful when eps >= lam*||v1||_inf).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .model import RelayPINN, u0_np
from .multifunction import dist2_relay_eps, branch_id


def _batched(model, pts, dtype, fn, batch=4096):
    outs = []
    for i in range(0, len(pts), batch):
        b = pts[i:i + batch]
        t = tf.convert_to_tensor(b[:, :1], dtype)
        x = tf.convert_to_tensor(b[:, 1:2], dtype)
        y = tf.convert_to_tensor(b[:, 2:3], dtype)
        outs.append(fn(t, x, y))
    return [np.concatenate([o[k].numpy() for o in outs], axis=0)
            for k in range(len(outs[0]))]


def evaluate_run(cfg: dict, run_dir: Path, reference: dict, out_json: Path) -> dict:
    tf.keras.backend.set_floatx(cfg.get("dtype", "float32"))
    dtype = tf.keras.backend.floatx()

    run_cfg = json.loads((run_dir / "run_config.json").read_text())
    lam, eps, T = float(run_cfg["lam"]), float(run_cfg["eps"]), float(cfg["T"])

    model = RelayPINN(width=cfg["width"], depth=cfg["depth"], seed=0)
    # build variables
    z0 = tf.zeros((1, 1), dtype)
    model.u(z0, z0, z0)
    model.load_weights_npz(run_dir / "weights.npz")

    t_star_ref = float(reference["t_star"])

    # ---------------- dense-grid residual, stratified -----------------
    rng = np.random.default_rng(777)
    M = int(cfg["eval_dense_points"])
    pts = np.concatenate([rng.uniform(0, T, (M, 1)),
                          rng.uniform(0, 1, (M, 1)),
                          rng.uniform(0, 1, (M, 1))], axis=1)

    def op_fn(t, x, y):
        u, z = model.operator(t, x, y)
        r = dist2_relay_eps(u, z, lam, eps)
        return u, z, r

    u_d, z_d, r_d = _batched(model, pts, dtype, op_fn)
    r_d = r_d[:, 0]
    tcol = pts[:, 0]
    strata = dict(
        pre=(tcol < 0.9 * t_star_ref),
        front=((tcol >= 0.9 * t_star_ref) & (tcol <= 1.1 * t_star_ref)),
        post=(tcol > 1.1 * t_star_ref),
    )
    residual = dict(mean=float(r_d.mean()), max=float(r_d.max()))
    for name, m in strata.items():
        residual[f"mean_{name}"] = float(r_d[m].mean()) if m.any() else None
        residual[f"max_{name}"] = float(r_d[m].max()) if m.any() else None

    # ---------------- norm curves and band-entry times ----------------
    nt = int(cfg["eval_nt"]); ng = int(cfg["eval_ng"])
    times = np.linspace(0.0, T, nt)
    s = np.linspace(0.0, 1.0, ng + 2)[1:-1]
    X, Y = np.meshgrid(s, s, indexing="ij")
    xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    h = s[1] - s[0]

    l2 = np.empty(nt); linf = np.empty(nt)
    uT = None
    for k, tk in enumerate(times):
        grid = np.concatenate([np.full((len(xy), 1), tk), xy], axis=1)
        (u_g,) = _batched(model, grid, dtype, lambda t, x, y: (model.u(t, x, y),))
        u_g = u_g[:, 0]
        l2[k] = h * np.linalg.norm(u_g)
        linf[k] = np.max(np.abs(u_g))
        if k == nt - 1:
            uT = u_g.reshape(ng, ng)

    inside = linf <= eps if eps > 0 else linf <= 1e-3  # eps=0: legacy threshold
    t_band = float(times[np.argmax(inside)]) if inside.any() else None
    t_band_persistent = None
    for k in range(nt):
        if inside[k:].all():
            t_band_persistent = float(times[k]); break

    # pre-extinction discrepancy vs reference norm curve
    ref_t = np.asarray(reference["times"]); ref_l2 = np.asarray(reference["l2"])
    mask = times < t_star_ref
    ref_interp = np.interp(times[mask], ref_t, ref_l2)
    norm_disc = float(np.max(np.abs(l2[mask] - ref_interp)) / np.max(ref_l2))

    # ---------------- branch-selection diagnostic ---------------------
    Nc = int(cfg["eval_cloud_points"])
    cloud_pts = np.concatenate([rng.uniform(0, T, (Nc, 1)),
                                rng.uniform(0, 1, (Nc, 1)),
                                rng.uniform(0, 1, (Nc, 1))], axis=1)
    u_c, z_c, r_c = _batched(model, cloud_pts, dtype, op_fn)
    tol = float(cfg["eval_violation_tol"])
    violations = int((r_c[:, 0] > tol).sum())

    # ---------------- non-uniqueness witness --------------------------
    v1 = np.asarray(reference["v1"])          # on reference interior grid
    v1_inf = float(reference["v1_inf"])
    # interpolate lam*v1 to the eval grid via bilinear-on-uniform (same box)
    from scipy.interpolate import RegularGridInterpolator
    n_ref = v1.shape[0]
    sr = np.linspace(0, 1, n_ref + 2)[1:-1]
    itp = RegularGridInterpolator((sr, sr), lam * v1, bounds_error=False,
                                  fill_value=0.0)
    wit = itp(xy).reshape(ng, ng)
    d_zero = float(h * np.linalg.norm(uT))
    d_wit = float(h * np.linalg.norm(uT - wit))
    witness_admissible = bool(eps >= lam * v1_inf)

    result = dict(
        run=run_dir.name, lam=lam, eps=eps, seed=run_cfg["seed"],
        sampler=run_cfg["sampler"],
        final_val_mean=run_cfg["final_val_mean"],
        final_val_max=run_cfg["final_val_max"],
        residual=residual,
        t_band=t_band, t_band_persistent=t_band_persistent,
        norm_discrepancy_pre=norm_disc,
        violations=violations, cloud_size=Nc, violation_tol=tol,
        tail_dist_to_zero=d_zero, tail_dist_to_witness=d_wit,
        witness_admissible=witness_admissible, lam_v1_inf=lam * v1_inf,
    )
    np.savez(run_dir / "eval_arrays.npz",
             times=times, l2=l2, linf=linf,
             cloud_s=u_c[:, 0], cloud_z=z_c[:, 0], cloud_r=r_c[:, 0],
             cloud_branch=np.asarray(branch_id(
                 tf.convert_to_tensor(u_c), eps)).reshape(-1))
    out_json.write_text(json.dumps(result, indent=2))
    return result
