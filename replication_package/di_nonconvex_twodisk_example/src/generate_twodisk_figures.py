#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.io as pio

T, N, K = 3.0, 120, 24
N_PHI = 120
t = np.linspace(0.0, T, N)
dt = t[1] - t[0]
tn = np.linspace(0.0, T, K)
x0 = np.array([0.0, 0.0])

R_DISK = 0.10
C_MAG = 0.40
theta = np.pi * t / T
C = C_MAG * np.stack([np.cos(theta), np.sin(theta)], axis=1)   # c(t_i)
GAP = C_MAG - R_DISK                                           # dist(0, E(t))


def g(ti: float, x: np.ndarray) -> np.ndarray:
    return np.array([np.sin(x[0]) + 0.15 * np.cos(2.0 * np.pi * ti / T),
                     0.5 * np.cos(x[1]) - 0.20 * np.sin(2.0 * np.pi * ti / T)])


def dist_to_E(xi_arr: np.ndarray) -> np.ndarray:
    d_plus = np.maximum(np.linalg.norm(xi_arr - C, axis=1) - R_DISK, 0.0)
    d_minus = np.maximum(np.linalg.norm(xi_arr + C, axis=1) - R_DISK, 0.0)
    return np.minimum(d_plus, d_minus)


def simulate(xi_arr: np.ndarray) -> np.ndarray:
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


def interp_nodes(Z: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, tn, Z[:, j]) for j in range(Z.shape[1])], axis=1)


NODE_OF = np.minimum(np.arange(N) * K // N, K - 1)

X_RELAXED = simulate(np.zeros((N, 2)))
X1 = X_RELAXED[-1].copy()

LAM_T, LAM_F, LAM_U = 2500.0, 1.0, 5e-2
LAMS_LOW, LAMS_HIGH = 2.0, 50.0
EPS = 1e-10


def xi_soft(z: np.ndarray) -> np.ndarray:
    return interp_nodes(z.reshape(K, 2))


def cost_soft(z: np.ndarray, lam_s: float) -> float:
    xi = xi_soft(z)
    X = simulate(xi)
    return (LAM_T * float(np.sum((X[-1] - X1) ** 2))
            + LAM_F * float(np.mean(dist_to_E(xi) ** 2))
            + lam_s * float(np.mean(np.sum(np.diff(xi, axis=0) ** 2, axis=1))))


def xi_hard(zc: np.ndarray, modes: np.ndarray) -> np.ndarray:
    W = zc[:2 * K].reshape(K, 2)
    RHO = zc[2 * K:]
    Wg = interp_nodes(W)
    rho_g = np.interp(t, tn, RHO)
    d = Wg / np.sqrt(np.sum(Wg ** 2, axis=1, keepdims=True) + EPS)
    s = 1.0 / (1.0 + np.exp(-rho_g))
    m = modes[NODE_OF][:, None]
    return m * C + (R_DISK * s)[:, None] * d


def cost_hard(zc: np.ndarray, modes: np.ndarray) -> float:
    xi = xi_hard(zc, modes)
    X = simulate(xi)
    W = zc[:2 * K].reshape(K, 2)
    RHO = zc[2 * K:]
    J_u = float(np.mean(np.sum(np.diff(W, axis=0) ** 2, axis=1)) + np.mean(np.diff(RHO) ** 2))
    return LAM_T * float(np.sum((X[-1] - X1) ** 2)) + LAM_U * J_u


def solve_all(cache: Path, recompute: bool) -> dict:
    if cache.exists() and not recompute:
        data = dict(np.load(cache))
        print(f"Loaded cached solution: {cache}")
        return data

    rng = np.random.default_rng(0)
    print("Solving (B1), (B2): soft distance-residual ...")
    soft = {}
    for lam_s in (LAMS_LOW, LAMS_HIGH):
        z0 = 0.05 * rng.standard_normal(2 * K)
        res = minimize(cost_soft, z0, args=(lam_s,), method="L-BFGS-B",
                       options=dict(maxiter=400, maxfun=200000))
        soft[lam_s] = xi_soft(res.x)

    print("Solving (A): hard selector (L-BFGS-B + greedy mode flips) ...")
    modes = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(K)])
    zc = 0.1 * rng.standard_normal(3 * K)
    for _ in range(3):
        res = minimize(cost_hard, zc, args=(modes,), method="L-BFGS-B",
                       options=dict(maxiter=400, maxfun=200000))
        zc = res.x
        base = res.fun
        for j in range(K):
            modes[j] *= -1.0
            cj = cost_hard(zc, modes)
            if cj < base:
                base = cj
            else:
                modes[j] *= -1.0

    data = dict(
        xi_H=xi_hard(zc, modes), modes=modes,
        xi_S1=soft[LAMS_LOW], xi_S2=soft[LAMS_HIGH],
    )
    data["X_H"] = simulate(data["xi_H"])
    data["X_S1"] = simulate(data["xi_S1"])
    data["X_S2"] = simulate(data["xi_S2"])
    np.savez(cache, **data)
    print(f"Cached solution: {cache}")
    return data


PAPER_BG = "white"
SCENE_BG = "white"
FONT_COLOR = "#111111"
GRID_COLOR = "rgba(0,0,0,0.16)"
ZERO_LINE = "rgba(0,0,0,0.28)"

TUBE_SCALE = "Blues"
SELECTOR_SCALE = "YlGnBu"
LEVEL_SCALE = "OrRd"
TIME_SCALE = "Viridis"

DRIFT_COLOR = "#1b7837"
VELOCITY_COLOR = "#b2182b"
CONNECTOR_COLOR = "rgba(60,60,60,0.65)"
STATE_COLOR = "#2166ac"
TARGET_COLOR = "#d95f02"
SOFT_COLOR = "#e08214"
HULL_COLOR = "Greys"


def scene_style(x_title: str, y_title: str, z_title: str = "t") -> dict:
    axis_common = dict(
        showbackground=True,
        backgroundcolor="white",
        gridcolor=GRID_COLOR,
        zerolinecolor=ZERO_LINE,
        linecolor="rgba(0,0,0,0.55)",
        tickfont=dict(color=FONT_COLOR, size=13),
        titlefont=dict(color=FONT_COLOR, size=15),
    )
    return dict(
        xaxis_title=x_title,
        yaxis_title=y_title,
        zaxis_title=z_title,
        bgcolor=SCENE_BG,
        xaxis=axis_common,
        yaxis=axis_common,
        zaxis=axis_common,
        camera=dict(eye=dict(x=1.65, y=-1.85, z=1.15)),
        aspectmode="cube",
    )


def base_layout(fig: go.Figure, scene: dict, width: int = 900, height: int = 700,
                right_margin: int = 0) -> None:
    fig.update_layout(
        template="plotly_white",
        title=None,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=SCENE_BG,
        font=dict(color=FONT_COLOR, family="Arial", size=14),
        scene=scene,
        width=width,
        height=height,
        margin=dict(l=0, r=right_margin, t=0, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.78)",
                    bordercolor="rgba(0,0,0,0.25)", borderwidth=1),
    )


def disk_tube_arrays(sign: float, centers: np.ndarray | None) -> tuple[np.ndarray, ...]:
    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI)
    X = np.zeros((N, N_PHI))
    Y = np.zeros((N, N_PHI))
    Z = np.zeros((N, N_PHI))
    for i in range(N):
        cx, cy = sign * C[i]
        if centers is not None:
            cx += centers[i, 0]
            cy += centers[i, 1]
        X[i] = cx + R_DISK * np.cos(phi)
        Y[i] = cy + R_DISK * np.sin(phi)
        Z[i] = t[i]
    return X, Y, Z


def stadium_hull_arrays(n_arc: int = 40, n_seg: int = 14) -> tuple[np.ndarray, ...]:
    M = 2 * n_arc + 2 * n_seg
    X = np.zeros((N, M))
    Y = np.zeros((N, M))
    Z = np.zeros((N, M))
    for i in range(N):
        u = C[i] / np.linalg.norm(C[i])
        nvec = np.array([-u[1], u[0]])
        pts = []
        for al in np.linspace(-np.pi / 2, np.pi / 2, n_arc):  # arc around +c
            d = np.cos(al) * u + np.sin(al) * nvec
            pts.append(C[i] + R_DISK * d)
        for sseg in np.linspace(0.0, 1.0, n_seg):     # tangent segment
            pts.append((1 - sseg) * (C[i] + R_DISK * nvec) + sseg * (-C[i] + R_DISK * nvec))
        for al in np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc):       # arc around -c
            d = np.cos(al) * u + np.sin(al) * nvec
            pts.append(-C[i] + R_DISK * d)
        for sseg in np.linspace(0.0, 1.0, n_seg):  # tangent segment
            pts.append((1 - sseg) * (-C[i] - R_DISK * nvec) + sseg * (C[i] - R_DISK * nvec))
        P = np.array(pts)
        X[i], Y[i], Z[i] = P[:, 0], P[:, 1], t[i]
    return X, Y, Z


def add_tube(fig: go.Figure, X, Y, Z, name: str, opacity: float = 0.36,
             colorscale: str = TUBE_SCALE, showlegend: bool = False) -> None:
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=opacity,
        colorscale=colorscale,
        showscale=False,
        name=name,
        showlegend=showlegend,
        hoverinfo="skip",
        lighting=dict(ambient=0.72, diffuse=0.55, specular=0.08, roughness=0.8),
    ))


def build_velocity_double_tube(data: dict) -> go.Figure:
    """F(t, x(t)) = g(t, x(t)) + [D(c,r) U D(-c,r)] along the HARD trajectory."""
    X_H, xi_H, modes = data["X_H"], data["xi_H"], data["modes"]
    G = np.array([g(t[i], X_H[i]) for i in range(N)])
    V = G + xi_H
    dist_H = dist_to_E(xi_H)

    fig = go.Figure()
    for sign, nm in ((+1.0, "g + c(t) + D(0,r)"), (-1.0, "g - c(t) + D(0,r)")):
        Xs, Ys, Zs = disk_tube_arrays(sign, centers=G)
        add_tube(fig, Xs, Ys, Zs, nm)

    fig.add_trace(go.Scatter3d(
        x=G[:, 0], y=G[:, 1], z=t,
        mode="lines+markers",
        name="center g(t,x(t))",
        marker=dict(size=3.8, color=DRIFT_COLOR),
        line=dict(width=5, color=DRIFT_COLOR),
        hovertemplate="t=%{z:.3f}<br>g=(%{x:.4f}, %{y:.4f})<extra></extra>",
    ))

    hover = [
        f"t={t[i]:.3f}<br>"
        f"mode m={int(modes[NODE_OF[i]]):+d}<br>"
        f"g=({G[i,0]:.4f}, {G[i,1]:.4f})<br>"
        f"xi=({xi_H[i,0]:.4f}, {xi_H[i,1]:.4f})<br>"
        f"v=({V[i,0]:.4f}, {V[i,1]:.4f})<br>"
        f"dist(xi,E)={dist_H[i]:.2e}"
        for i in range(N)
    ]
    fig.add_trace(go.Scatter3d(
        x=V[:, 0], y=V[:, 1], z=t,
        mode="markers",
        name="selected velocities v(t_i)=g+xi (hard)",
        marker=dict(size=5.0, color=VELOCITY_COLOR),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    connector_idx = list(range(0, N, 8))
    if connector_idx[-1] != N - 1:
        connector_idx.append(N - 1)
    for j, i in enumerate(connector_idx):
        fig.add_trace(go.Scatter3d(
            x=[G[i, 0], V[i, 0]], y=[G[i, 1], V[i, 1]], z=[t[i], t[i]],
            mode="lines",
            name="xi(t_i)=v-g" if j == 0 else None,
            showlegend=(j == 0),
            line=dict(width=4, color=CONNECTOR_COLOR),
            hoverinfo="skip",
        ))

    base_layout(fig, scene_style("component 1", "component 2", "t"))
    return fig


def build_selector_double_cylinder(data: dict) -> go.Figure:
    """xi(t) inside E(t) = two rotating disk cylinders; hard vs soft selectors."""
    xi_H, xi_S1, xi_S2, modes = data["xi_H"], data["xi_S1"], data["xi_S2"], data["modes"]
    d_H, d_S1, d_S2 = dist_to_E(xi_H), dist_to_E(xi_S1), dist_to_E(xi_S2)

    fig = go.Figure()

    Xh, Yh, Zh = stadium_hull_arrays()
    fig.add_trace(go.Surface(
        x=Xh, y=Yh, z=Zh,
        opacity=0.10,
        colorscale=HULL_COLOR,
        showscale=False,
        name="conv E(t) (stadium hull)",
        showlegend=True,
        hoverinfo="skip",
        lighting=dict(ambient=0.85, diffuse=0.3, specular=0.02, roughness=0.95),
    ))

    for sign in (+1.0, -1.0):
        Xs, Ys, Zs = disk_tube_arrays(sign, centers=None)
        add_tube(fig, Xs, Ys, Zs, f"E(t): disk at {'+' if sign > 0 else '-'}c(t)",
                 opacity=0.42, colorscale=SELECTOR_SCALE)

    for sign, dash in ((+1.0, "solid"), (-1.0, "solid")):
        fig.add_trace(go.Scatter3d(
            x=sign * C[:, 0], y=sign * C[:, 1], z=t,
            mode="lines",
            name="disk centers ±c(t)" if sign > 0 else None,
            showlegend=(sign > 0),
            line=dict(width=2, color="rgba(0,0,0,0.35)", dash=dash),
            hoverinfo="skip",
        ))

    hov_H = [
        f"t={t[i]:.3f}<br>mode m={int(modes[NODE_OF[i]]):+d}<br>"
        f"xi=({xi_H[i,0]:.4f}, {xi_H[i,1]:.4f})<br>dist(xi,E)={d_H[i]:.2e}"
        for i in range(N)
    ]
    fig.add_trace(go.Scatter3d(
        x=xi_H[:, 0], y=xi_H[:, 1], z=t,
        mode="lines",
        name="line through xi(t_i) (hard)",
        line=dict(width=2, color="rgba(0,0,0,0.30)"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=xi_H[:, 0], y=xi_H[:, 1], z=t,
        mode="markers",
        name="hard selector xi(t_i)  [dist=0]",
        marker=dict(size=4.6, color=DRIFT_COLOR),
        text=hov_H,
        hovertemplate="%{text}<extra></extra>",
    ))

    hov_S2 = [
        f"t={t[i]:.3f}<br>xi=({xi_S2[i,0]:.4f}, {xi_S2[i,1]:.4f})<br>"
        f"dist(xi,E)={d_S2[i]:.4f}"
        for i in range(N)
    ]
    fig.add_trace(go.Scatter3d(
        x=xi_S2[:, 0], y=xi_S2[:, 1], z=t,
        mode="lines",
        name="line through xi(t_i) (soft, high smoothing)",
        line=dict(width=2, color="rgba(178,24,43,0.45)"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=xi_S2[:, 0], y=xi_S2[:, 1], z=t,
        mode="markers",
        name="soft DR selector (relaxed)  [color: dist]",
        marker=dict(
            size=5.2,
            color=d_S2,
            cmin=0.0,
            cmax=GAP,
            colorscale=LEVEL_SCALE,
            showscale=True,
            colorbar=dict(title="dist(xi,E)", x=1.02,
                          tickfont=dict(color=FONT_COLOR),
                          titlefont=dict(color=FONT_COLOR)),
        ),
        text=hov_S2,
        hovertemplate="%{text}<extra></extra>",
    ))

    hov_S1 = [
        f"t={t[i]:.3f}<br>xi=({xi_S1[i,0]:.4f}, {xi_S1[i,1]:.4f})<br>"
        f"dist(xi,E)={d_S1[i]:.4f}"
        for i in range(N)
    ]
    fig.add_trace(go.Scatter3d(
        x=xi_S1[:, 0], y=xi_S1[:, 1], z=t,
        mode="markers",
        name="soft DR selector (chattering) — click to show",
        visible="legendonly",
        marker=dict(size=4.6, color=d_S1, cmin=0.0, cmax=GAP,
                    colorscale=LEVEL_SCALE, showscale=False),
        text=hov_S1,
        hovertemplate="%{text}<extra></extra>",
    ))

    base_layout(fig, scene_style("xi_1", "xi_2", "t"), right_margin=90)
    return fig


def build_state_trajectories(data: dict) -> go.Figure:
    X_H, X_S1, X_S2 = data["X_H"], data["X_S1"], data["X_S2"]
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X_RELAXED[:, 0], y=X_RELAXED[:, 1], z=t,
        mode="lines",
        name="relaxed reference (xi=0)",
        line=dict(width=4, color="rgba(0,0,0,0.45)", dash="dash"),
        hovertemplate="t=%{z:.3f}<br>x=(%{x:.4f}, %{y:.4f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=X_H[:, 0], y=X_H[:, 1], z=t,
        mode="lines+markers",
        name="(A) hard selector",
        line=dict(width=6, color=STATE_COLOR),
        marker=dict(size=3.6, color=t, colorscale=TIME_SCALE, showscale=True,
                    colorbar=dict(title="t", x=1.02, tickfont=dict(color=FONT_COLOR),
                                  titlefont=dict(color=FONT_COLOR))),
        hovertemplate="t=%{z:.3f}<br>x=(%{x:.4f}, %{y:.4f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=X_S1[:, 0], y=X_S1[:, 1], z=t,
        mode="lines",
        name="(B1) soft DR, low smoothing",
        line=dict(width=4, color=VELOCITY_COLOR),
        hovertemplate="t=%{z:.3f}<br>x=(%{x:.4f}, %{y:.4f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=X_S2[:, 0], y=X_S2[:, 1], z=t,
        mode="lines",
        name="(B2) soft DR, high smoothing (relaxed)",
        line=dict(width=4, color=SOFT_COLOR),
        hovertemplate="t=%{z:.3f}<br>x=(%{x:.4f}, %{y:.4f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[x0[0]], y=[x0[1]], z=[0.0],
        mode="markers", name="start",
        marker=dict(size=9, color=DRIFT_COLOR, symbol="circle"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[X1[0]], y=[X1[1]], z=[T],
        mode="markers", name="target x(T)",
        marker=dict(size=10, color=TARGET_COLOR, symbol="diamond"),
    ))

    base_layout(fig, scene_style("x_1", "x_2", "t"), height=650, right_margin=65)
    return fig


def write_html(fig: go.Figure, path: Path, include_plotlyjs: str | bool = "cdn") -> None:
    pio.write_html(fig, file=str(path), include_plotlyjs=include_plotlyjs, full_html=True)


def write_static(fig: go.Figure, path: Path, scale: int) -> None:
    try:
        pio.write_image(fig, str(path), scale=scale)
    except Exception as exc:
        raise RuntimeError(
            f"Could not export {path.name}. Static export requires kaleido. "
            "Run `python -m pip install -r requirements.txt`. "
            "If you use a newer Kaleido that asks for Chrome, pinning kaleido==0.2.1 usually avoids that requirement."
        ) from exc


def write_combined_html(figs: list[tuple[str, go.Figure]], path: Path) -> None:
    parts = []
    for i, (_, fig) in enumerate(figs):
        parts.append(pio.to_html(fig, include_plotlyjs="cdn" if i == 0 else False, full_html=False))
    body = "\n<hr style='border:0;border-top:1px solid #ddd;margin:30px 0;'>\n".join(parts)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Two-disk nonconvex inclusion: publication figures</title>
<style>
body {{ margin: 0; padding: 24px; background: white; color: #111; font-family: Arial, sans-serif; }}
.wrap {{ max-width: 980px; margin: 0 auto; }}
h1 {{ font-weight: 500; font-size: 24px; margin: 0 0 8px; }}
p {{ color: #444; line-height: 1.5; }}
</style>
</head>
<body>
<div class="wrap">
<h1>Two-disk nonconvex differential inclusion: publication figures</h1>
<p>Light-background Plotly figures for manuscript/PDF use. Hard selector (admissible by
construction, chattering between disks) vs soft distance residual (relaxed selection in
conv E(t)). Use the PNG files for the final LaTeX build when possible.</p>
{body}
</div>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Plotly two-disk inclusion figures.")
    parser.add_argument("--out", default="outputs", help="Output directory. Default: outputs")
    parser.add_argument("--scale", type=int, default=3, help="Static export scale. Default: 3")
    parser.add_argument("--formats", default="png,pdf,html",
                        help="Comma-separated list from: png,pdf,html. Default: png,pdf,html")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cached optimization results and re-solve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    allowed = {"png", "pdf", "html"}
    unknown = formats - allowed
    if unknown:
        raise SystemExit(f"Unknown format(s): {', '.join(sorted(unknown))}. Allowed: png,pdf,html")

    data = solve_all(out_dir / "twodisk_solution.npz", recompute=args.recompute)

    for nm, xi in (("hard", data["xi_H"]), ("soft-low", data["xi_S1"]), ("soft-high", data["xi_S2"])):
        d = dist_to_E(xi)
        Xend = simulate(xi)[-1]
        print(f"  {nm:10s} |x(T)-x1|={np.linalg.norm(Xend - X1):.2e}  "
              f"mean dist={d.mean():.3e}  max dist={d.max():.3e}")

    figs = [
        ("01_velocity_double_tube_light", build_velocity_double_tube(data)),
        ("02_selector_double_cylinder_light", build_selector_double_cylinder(data)),
        ("03_state_trajectories_light", build_state_trajectories(data)),
    ]
    for name, fig in figs:
        if "html" in formats:
            write_html(fig, out_dir / f"{name}.html")
        if "png" in formats:
            write_static(fig, out_dir / f"{name}.png", scale=args.scale)
        if "pdf" in formats:
            write_static(fig, out_dir / f"{name}.pdf", scale=args.scale)

    if "html" in formats:
        write_combined_html(figs, out_dir / "combined_light_figures.html")

    print(f"Wrote outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
