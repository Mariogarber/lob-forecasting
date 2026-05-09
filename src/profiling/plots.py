# src/profiling/plots.py
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def grid_axes(
    n_panels: int,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (8.0, 4.0),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create a grid of axes with at least n_panels cells.

    Returns (fig, axes_flat) where axes_flat is a 1-D list of all axes,
    including any unused trailing ones which are hidden automatically.
    """
    nrows = math.ceil(n_panels / ncols)
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    fig, axes_2d = plt.subplots(nrows, ncols, figsize=figsize)

    if n_panels == 1:
        axes_flat = [axes_2d]
    elif nrows == 1:
        axes_flat = list(axes_2d)
    else:
        axes_flat = [ax for row in axes_2d for ax in row]

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    return fig, axes_flat
