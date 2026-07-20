import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

T, N, K = 3.0, 120, 24
t = np.linspace(0.0, T, N)
dt = t[1] - t[0]
tn = np.linspace(0.0, T, K)                      # node times
x0 = np.array([0.0, 0.0])

r_disk = 0.10
c_mag = 0.40
theta = np.pi * t / T
C = c_mag * np.stack([np.cos(theta), np.sin(theta)], axis=1)   # c(t_i), (N,2)

def g(ti, x):
    """Nonlinear time-forced drift (same as the rotating-ellipse example)."""
    return np.array([np.sin(x[0]) + 0.15 * np.cos(2.0 * np.pi * ti / T),
                     0.5 * np.cos(x[1]) - 0.20 * np.sin(2.0 * np.pi * ti / T)])

def dist_to_E(xi_arr):
    """dist(xi_i, E(t_i)) for xi_arr of shape (N,2), E = union of two disks
    (closed-form; no QP / Quickhull needed)."""
    d_plus = np.maximum(np.linalg.norm(xi_arr - C, axis=1) - r_disk, 0.0)
    d_minus = np.maximum(np.linalg.norm(xi_arr + C, axis=1) - r_disk, 0.0)
    return np.minimum(d_plus, d_minus)

def simulate(xi_arr):
    X = np.zeros((N, 2))
    X[0] = x0
    for i in range(N - 1):
        xi = xi_arr[i]
        x, ti = X[i], t[i]
        k1 = g(ti, x) + xi
        k2 = g(ti + dt / 2, x + dt / 2 * k1) + xi
        k3 = g(ti + dt / 2, x + dt / 2 * k2) + xi
        k4 = g(ti + dt, x + dt * k3) + xi
        X[i + 1] = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X

X_relaxed = simulate(np.zeros((N, 2)))
x1 = X_relaxed[-1].copy()
gap = c_mag - r_disk   # dist(0, E(t)) = 0.3, the "relaxation gap" scale

def interp_nodes(Z):
    return np.stack([np.interp(t, tn, Z[:, j]) for j in range(Z.shape[1])], axis=1)

node_of = np.minimum(np.arange(N) * K // N, K - 1)

lamT, lamF = 2500.0, 1.0
LAMS_LOW, LAMS_HIGH = 2.0, 50.0

def xi_soft(z):
    return interp_nodes(z.reshape(K, 2))

def cost_soft(z, lamS):
    xi = xi_soft(z)
    X = simulate(xi)
    J_term = np.sum((X[-1] - x1) ** 2)
    J_incl = np.mean(dist_to_E(xi) ** 2)
    J_smooth = np.mean(np.sum(np.diff(xi, axis=0) ** 2, axis=1))
    return lamT * J_term + lamF * J_incl + lamS * J_smooth

lamU = 5e-2
EPS = 1e-10

def xi_hard(zc, modes):
    W = zc[:2 * K].reshape(K, 2)
    RHO = zc[2 * K:]
    Wg = interp_nodes(W)
    rho_g = np.interp(t, tn, RHO)
    d = Wg / np.sqrt(np.sum(Wg ** 2, axis=1, keepdims=True) + EPS)
    s = 1.0 / (1.0 + np.exp(-rho_g))
    m = modes[node_of][:, None]
    return m * C + (r_disk * s)[:, None] * d

def cost_hard(zc, modes):
    xi = xi_hard(zc, modes)
    X = simulate(xi)
    J_term = np.sum((X[-1] - x1) ** 2)
    W = zc[:2 * K].reshape(K, 2)
    RHO = zc[2 * K:]
    J_u = np.mean(np.sum(np.diff(W, axis=0) ** 2, axis=1)) + np.mean(np.diff(RHO) ** 2)
    return lamT * J_term + lamU * J_u

print("=" * 72)
print("Two-disk nonconvex inclusion, target = relaxed (xi=0) endpoint")
print(f"T={T}, N={N}, K={K}, r={r_disk}, |c|={c_mag}, dist(0,E)={gap:.3f}")
print(f"target x1 = ({x1[0]:.4f}, {x1[1]:.4f})")
print("=" * 72)

soft = {}
for lamS in (LAMS_LOW, LAMS_HIGH):
    z0 = 0.05 * rng.standard_normal(2 * K)
    res = minimize(cost_soft, z0, args=(lamS,), method="L-BFGS-B",
                   options=dict(maxiter=400, maxfun=200000))
    xi = xi_soft(res.x)
    soft[lamS] = dict(xi=xi, X=simulate(xi), dist=dist_to_E(xi))
xi_S,  X_S,  dist_S  = (soft[LAMS_LOW][k]  for k in ("xi", "X", "dist"))
xi_S2, X_S2, dist_S2 = (soft[LAMS_HIGH][k] for k in ("xi", "X", "dist"))

modes = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(K)])  # chattering init
zc = 0.1 * rng.standard_normal(3 * K)
for rnd in range(3):
    res_hard = minimize(cost_hard, zc, args=(modes,), method="L-BFGS-B",
                        options=dict(maxiter=400, maxfun=200000))
    zc = res_hard.x
    base = res_hard.fun
    n_flips = 0
    for j in range(K):
        modes[j] *= -1.0
        cj = cost_hard(zc, modes)
        if cj < base:
            base = cj
            n_flips += 1
        else:
            modes[j] *= -1.0
    print(f"[hard] round {rnd + 1}: cost={base:.3e}, mode flips accepted={n_flips}")
xi_H = xi_hard(zc, modes)
X_H = simulate(xi_H)
dist_H = dist_to_E(xi_H)

def report(name, X, d):
    print(f"{name:28s} |x(T)-x1| = {np.linalg.norm(X[-1] - x1):.3e}   "
          f"mean dist(xi,E) = {d.mean():.3e}   max = {d.max():.3e}")

print("-" * 72)
report("(A) hard selector", X_H, dist_H)
report(f"(B1) soft DR, lamS={LAMS_LOW}", X_S, dist_S)
report(f"(B2) soft DR, lamS={LAMS_HIGH}", X_S2, dist_S2)
print("-" * 72)
print("(A)  admissible BY CONSTRUCTION: dist identically 0 up to eps.")
print("(B1) expressive surrogate: chatters, but every disk-to-disk crossing")
print(f"     spikes to ~the gap ({gap:.2f}); residual -> 0 only in relaxed sense.")
print("(B2) smooth surrogate: cannot chatter, so it selects a RELAXED solution")
print("     spending a positive fraction of time strictly inside the gap;")
print(f"     the residual acquires a positive floor with plateaus at ~{gap:.2f}.")
print("     Which relaxed solution is picked is optimizer-dependent")
print("     (non-uniqueness, cf. their Remark 4.5).")
print("Headline: mean residual (A) = 0 exactly vs (B1),(B2) >> 0 -- the")
print("distance-residual loss certifies only conv E(t) (Filippov-Wazewski).")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

a = ax[0, 0]
a.plot(X_relaxed[:, 0], X_relaxed[:, 1], "k--", lw=1, label="relaxed ref (xi=0)")
a.plot(X_H[:, 0], X_H[:, 1], color="tab:blue", lw=1.6, label="(A) hard selector")
a.plot(X_S[:, 0], X_S[:, 1], color="tab:red", lw=1.6, ls=":", label="(B1) soft, low $\\lambda_S$")
a.plot(X_S2[:, 0], X_S2[:, 1], color="tab:orange", lw=1.6, ls="-.", label="(B2) soft, high $\\lambda_S$")
a.plot(*x0, "ko", ms=6); a.plot(*x1, "kD", ms=7)
a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$")
a.set_title("state trajectories (start o, target $\\diamond$)")
a.legend(fontsize=8)

a = ax[0, 1]
a.plot(t, dist_S, color="tab:red", lw=1.4, label="(B1) soft, low $\\lambda_S$ (spikes at crossings)")
a.plot(t, dist_S2, color="tab:orange", lw=1.6, label="(B2) soft, high $\\lambda_S$ (relaxed plateau)")
a.plot(t, dist_H, color="tab:blue", lw=1.6, label="(A) hard selector ($\\equiv 0$)")
a.axhline(gap, color="k", ls="--", lw=1, label=f"relaxation gap $|c|-r$ = {gap}")
a.set_xlabel("$t$"); a.set_ylabel("dist$(\\xi(t_i), E(t_i))$")
a.set_title("pointwise inclusion residual (the gap)")
a.legend(fontsize=7)

a = ax[1, 0]
a.plot(t, xi_H[:, 0], color="tab:blue", lw=1.2, label="(A) $\\xi_1$")
a.plot(t, xi_H[:, 1], color="tab:blue", lw=1.2, ls="--", label="(A) $\\xi_2$")
a.plot(t, xi_S2[:, 0], color="tab:orange", lw=1.4, label="(B2) $\\xi_1$")
a.plot(t, xi_S2[:, 1], color="tab:orange", lw=1.4, ls="--", label="(B2) $\\xi_2$")
a.plot(t, C[:, 0], color="0.6", lw=0.8)
a.plot(t, -C[:, 0], color="0.6", lw=0.8)
a.set_xlabel("$t$"); a.set_ylabel("$\\xi$ components")
a.set_title("selector: exact chattering (A) vs smooth relaxed selection (B2)")
a.legend(fontsize=7, ncol=2)

a = ax[1, 1]
a.step(tn, modes, where="post", color="tab:blue", lw=1.5)
a.set_yticks([-1, 1]); a.set_ylim(-1.5, 1.5)
a.set_xlabel("$t$ (node grid)"); a.set_ylabel("disk mode $m_j$")
a.set_title("(A) optimized disk-selection sequence")

fig.suptitle("Nonconvex $E(t)$ = two rotating disks: hard selector vs distance residual", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("twodisk_relaxation_gap.png", dpi=150)
print("figure saved: twodisk_relaxation_gap.png")
