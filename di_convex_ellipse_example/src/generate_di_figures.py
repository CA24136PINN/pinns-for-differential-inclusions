#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


T = 3.0
N = 120
N_PHI = 120

t = np.linspace(0.0, T, N)
s = t / T

a = 0.30 + 0.40 * np.abs(np.sin(np.pi * s))
b = 0.15 + 0.25 * np.abs(np.cos(np.pi * s))
theta = np.pi * s

g1 = 0.15 + 0.78 * s + 0.10 * np.sin(2.0 * np.pi * s + 0.15)
g2 = 0.72 - 0.55 * s + 0.12 * np.cos(1.6 * np.pi * s)
g = np.column_stack([g1, g2])

# Selector xi(t), constructed inside E(t)
phi_sel = 1.25 * np.pi * s + 0.55 * np.sin(2.0 * np.pi * s)
rho = 0.45 + 0.38 * np.sin(1.35 * np.pi * s + 0.35) ** 2

u = rho * a * np.cos(phi_sel)
w = rho * b * np.sin(phi_sel)
level = (u / a) ** 2 + (w / b) ** 2

ct = np.cos(theta)
st = np.sin(theta)
xi = np.column_stack([ct * u - st * w, st * u + ct * w])
v = g + xi

# State trajectory induced by v(t), only for the auxiliary space-time plot
x = np.zeros((N, 2))
x[0] = np.array([0.02, -0.03])
dt = t[1] - t[0]
for k in range(1, N):
    x[k] = x[k - 1] + 0.42 * dt * v[k - 1]


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


def ellipse_tube_arrays(centered_at_g: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X,Y,Z,C arrays for a rotating ellipse tube."""
    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI)
    X = np.zeros((N, N_PHI))
    Y = np.zeros((N, N_PHI))
    Z = np.zeros((N, N_PHI))
    C = np.zeros((N, N_PHI))

    for i in range(N):
        ex = a[i] * np.cos(phi)
        ey = b[i] * np.sin(phi)
        rx = np.cos(theta[i]) * ex - np.sin(theta[i]) * ey
        ry = np.sin(theta[i]) * ex + np.cos(theta[i]) * ey

        if centered_at_g:
            X[i] = g[i, 0] + rx
            Y[i] = g[i, 1] + ry
        else:
            X[i] = rx
            Y[i] = ry

        Z[i] = t[i]
        C[i] = a[i] / b[i]

    return X, Y, Z, C


def velocity_hover() -> list[str]:
    return [
        f"t={t[i]:.3f}<br>"
        f"g=({g[i,0]:.4f}, {g[i,1]:.4f})<br>"
        f"xi=({xi[i,0]:.4f}, {xi[i,1]:.4f})<br>"
        f"v=({v[i,0]:.4f}, {v[i,1]:.4f})<br>"
        f"level={level[i]:.6f}<br>"
        f"a={a[i]:.3f}, b={b[i]:.3f}, theta={np.degrees(theta[i]):.1f} deg"
        for i in range(N)
    ]


def selector_hover() -> list[str]:
    return [
        f"t={t[i]:.3f}<br>"
        f"xi=({xi[i,0]:.4f}, {xi[i,1]:.4f})<br>"
        f"level={level[i]:.6f}<br>"
        f"a={a[i]:.3f}, b={b[i]:.3f}<br>"
        f"theta={np.degrees(theta[i]):.1f} deg"
        for i in range(N)
    ]


def build_velocity_tube_figure(connect_v_samples: bool = False, connector_every: int = 8) -> go.Figure:
    """3D tube F(t,x(t)) = g(t,x(t)) + E(t)."""
    X, Y, Z, _ = ellipse_tube_arrays(centered_at_g=True)
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        opacity=0.36,
        colorscale=TUBE_SCALE,
        showscale=False,
        name="F(t,x(t)) = g(t,x(t)) + E(t)",
        hoverinfo="skip",
        lighting=dict(ambient=0.72, diffuse=0.55, specular=0.08, roughness=0.8),
    ))

    fig.add_trace(go.Scatter3d(
        x=g[:, 0], y=g[:, 1], z=t,
        mode="lines+markers",
        name="center g(t,x(t))",
        marker=dict(size=3.8, color=DRIFT_COLOR),
        line=dict(width=5, color=DRIFT_COLOR),
        hovertemplate="t=%{z:.3f}<br>g=(%{x:.4f}, %{y:.4f})<extra></extra>",
    ))

    if connect_v_samples:
        fig.add_trace(go.Scatter3d(
            x=v[:, 0], y=v[:, 1], z=t,
            mode="lines",
            name="line through v(t_i)",
            line=dict(width=2, color=VELOCITY_COLOR),
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=v[:, 0], y=v[:, 1], z=t,
        mode="markers",
        name="selected velocities v(t_i)=g+xi",
        marker=dict(size=5.5, color=VELOCITY_COLOR),
        text=velocity_hover(),
        hovertemplate="%{text}<extra></extra>",
    ))

    connector_idx = list(range(0, N, connector_every))
    if connector_idx[-1] != N - 1:
        connector_idx.append(N - 1)

    for j, i in enumerate(connector_idx):
        fig.add_trace(go.Scatter3d(
            x=[g[i, 0], v[i, 0]],
            y=[g[i, 1], v[i, 1]],
            z=[t[i], t[i]],
            mode="lines",
            name="xi(t_i)=v-g" if j == 0 else None,
            showlegend=(j == 0),
            line=dict(width=4, color=CONNECTOR_COLOR),
            hoverinfo="skip",
        ))

    fig.update_layout(
        template="plotly_white",
        title=None,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=SCENE_BG,
        font=dict(color=FONT_COLOR, family="Arial", size=14),
        scene=scene_style("component 1", "component 2", "t"),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.78)", bordercolor="rgba(0,0,0,0.25)", borderwidth=1),
    )
    return fig


def build_selector_cylinder_figure(show_line: bool = True) -> go.Figure:
    """Selector cylinder xi(t) in E(t)."""
    X, Y, Z, C = ellipse_tube_arrays(centered_at_g=False)
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=C,
        colorscale=SELECTOR_SCALE,
        opacity=0.42,
        showscale=True,
        colorbar=dict(title="a/b", x=0.98, tickfont=dict(color=FONT_COLOR), titlefont=dict(color=FONT_COLOR)),
        name="admissible ellipse cylinder E(t)",
        hoverinfo="skip",
        lighting=dict(ambient=0.76, diffuse=0.5, specular=0.06, roughness=0.85),
    ))

    if show_line:
        fig.add_trace(go.Scatter3d(
            x=xi[:, 0], y=xi[:, 1], z=t,
            mode="lines",
            name="line through xi(t_i)",
            line=dict(width=2, color="rgba(0,0,0,0.46)"),
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter3d(
        x=xi[:, 0], y=xi[:, 1], z=t,
        mode="markers",
        name="selector values xi(t_i)",
        marker=dict(
            size=5.5,
            color=level,
            cmin=0,
            cmax=1,
            colorscale=LEVEL_SCALE,
            showscale=True,
            colorbar=dict(title="level", x=1.10, tickfont=dict(color=FONT_COLOR), titlefont=dict(color=FONT_COLOR)),
        ),
        text=selector_hover(),
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_white",
        title=None,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=SCENE_BG,
        font=dict(color=FONT_COLOR, family="Arial", size=14),
        scene=scene_style("xi_1", "xi_2", "t"),
        width=900,
        height=700,
        margin=dict(l=0, r=90, t=0, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.78)", bordercolor="rgba(0,0,0,0.25)", borderwidth=1),
    )
    return fig


def build_state_trajectory_figure() -> go.Figure:
    """State-space trajectory x(t)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x[:, 0], y=x[:, 1], z=t,
        mode="lines+markers",
        name="trajectory x(t)",
        line=dict(width=6, color=STATE_COLOR),
        marker=dict(size=4.5, color=t, colorscale=TIME_SCALE, showscale=True, colorbar=dict(title="t")),
        hovertemplate="t=%{z:.3f}<br>x=(%{x:.4f}, %{y:.4f})<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[0, 0]], y=[x[0, 1]], z=[0.0],
        mode="markers", name="start",
        marker=dict(size=9, color=DRIFT_COLOR, symbol="circle"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[-1, 0]], y=[x[-1, 1]], z=[T],
        mode="markers", name="target x(T)",
        marker=dict(size=10, color=TARGET_COLOR, symbol="diamond"),
    ))
    fig.update_layout(
        template="plotly_white",
        title=None,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=SCENE_BG,
        font=dict(color=FONT_COLOR, family="Arial", size=14),
        scene=scene_style("x_1", "x_2", "t"),
        width=900,
        height=650,
        margin=dict(l=0, r=65, t=0, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.78)", bordercolor="rgba(0,0,0,0.25)", borderwidth=1),
    )
    return fig


def figure_specs() -> list[tuple[str, go.Figure]]:
    return [
        ("01_velocity_tube_light", build_velocity_tube_figure(connect_v_samples=False)),
        ("02_selector_cylinder_light", build_selector_cylinder_figure(show_line=True)),
        ("03_state_trajectory_light", build_state_trajectory_figure()),
    ]


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
<title>Differential inclusion publication figures</title>
<style>
body {{ margin: 0; padding: 24px; background: white; color: #111; font-family: Arial, sans-serif; }}
.wrap {{ max-width: 980px; margin: 0 auto; }}
h1 {{ font-weight: 500; font-size: 24px; margin: 0 0 8px; }}
p {{ color: #444; line-height: 1.5; }}
</style>
</head>
<body>
<div class="wrap">
<h1>Differential inclusion publication figures</h1>
<p>Light-background Plotly figures for manuscript/PDF use. Use the PNG files for the final LaTeX build when possible.</p>
{body}
</div>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Plotly differential-inclusion figures.")
    parser.add_argument("--out", default="outputs", help="Output directory. Default: outputs")
    parser.add_argument("--scale", type=int, default=3, help="Static export scale. Default: 3")
    parser.add_argument(
        "--formats",
        default="png,pdf,html",
        help="Comma-separated list from: png,pdf,html. Default: png,pdf,html",
    )
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

    figs = figure_specs()
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
