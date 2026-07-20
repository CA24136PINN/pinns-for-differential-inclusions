# Differential inclusion Plotly publication figures

This package generates publication-style Plotly figures:

1. `01_velocity_tube_light` — twisting ellipse tube `F(t,x(t)) = g(t,x(t)) + E(t)`.
2. `02_selector_cylinder_light` — selector `xi(t)` inside the rotating ellipse `E(t)`.
3. `03_state_trajectory_light` — auxiliary state trajectory `x(t)`.

The script exports each figure as:

- `.png` — recommended for LaTeX/PDF submission, because Plotly 3D/WebGL PDF export is rasterized internally;
- `.pdf` — convenient for direct inclusion/checking;
- `.html` — interactive Plotly version.

The folder `outputs/` already contains a pre-generated set of PNG/PDF files.

## Quick run

```bash
chmod +x run_export.sh
./run_export.sh # or bash run_export.sh
```

The output files will be written to `outputs/`.

## Custom output directory and scale

```bash
./run_export.sh my_outputs 4
```

Here `4` is the static image export scale. Larger values produce larger PNG/PDF rasters.

## Manual run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/generate_di_figures.py --out outputs --scale 3 --formats png,pdf,html
```
