"""Shared matplotlib style for all DR-PINN paper figures.

Every experiment notebook must call ``apply_paper_style()`` right after
importing matplotlib, so that all figures in the manuscript share the same
typography (LaTeX-like Computer Modern at the paper's 14pt body size) and
the same export resolution (dpi=180).

Usage
-----
    import matplotlib.pyplot as plt
    from paper_style import apply_paper_style
    apply_paper_style()
"""

import matplotlib.pyplot as plt

#: Export resolution used for every figure included in the manuscript.
PAPER_DPI = 180

PAPER_RC = {
    # LaTeX-like Computer Modern text at the paper's body-text size (14pt)
    "mathtext.fontset":            "cm",
    "font.family":                 "serif",
    "font.serif":                  ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "font.size":                   14,
    "axes.labelsize":              14,
    "axes.titlesize":              14,
    "xtick.labelsize":             14,
    "ytick.labelsize":             14,
    "legend.fontsize":             14,
    # Uniform export settings
    "savefig.dpi":                 PAPER_DPI,
    "savefig.bbox":                "tight",
}


def apply_paper_style() -> None:
    """Apply the shared paper style to the current matplotlib session."""
    plt.rcParams.update(PAPER_RC)
