import os

import matplotlib.pyplot as plt

# Every figure writes here, relative to scripts/.
OUTPUT_DIR = "output"

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": "Nimbus Sans",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLORS = {
    # Split contrast -- fixed meaning across the whole paper.
    "primary": "#0072B2",       # temporal split / the main series
    "secondary": "#D55E00",     # random split / the contrast series
    "baseline": "#222222",      # reference lines (median, threshold)
    "neutral": "#666666",       # annotations, secondary axes
    # Distribution figures: muted bars so an overlaid fit stays readable.
    "hist": "#9EC5E8",
    "hist_edge": "#4A7FA8",
    "fit": "#D55E00",           # the fitted total
    "band": "#CC79A7",          # highlighted interval (e.g. the 30 min spike)
    # Kept for the failure-vs-feature panels, which name colours semantically.
    "danger": "#D55E00",
    "default_line": "#0072B2",
    "volume_fill": "#56B4E9",
    "bar_fill": "#4A7FA8",
    "gridline": "#EEEEEE",   # heatmap cell separators
}

# Mixture / exponential components, in component order. Distinct from COLORS["fit"]
# so a component curve is never confused with the total.
COMPONENTS = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# Categorical cycle for site / experiment / job-type overlays.
SERIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
          "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def series_colors(n):
    """n categorical colours, cycling SERIES when there are more levels than
    entries. Site and experiment overlays routinely exceed the palette."""
    return [SERIES[i % len(SERIES)] for i in range(n)]


def apply_style():
    """Install the paper rcParams. Call once at the top of a figure cell."""
    plt.rcParams.update(_RC)


def save_fig(fig, name, ext="pdf"):
    """Write <OUTPUT_DIR>/<name>.<ext>. Returns the path so a cell can print it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.{ext}")
    fig.savefig(path, bbox_inches="tight")
    return path
