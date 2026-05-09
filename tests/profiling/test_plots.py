# tests/profiling/test_plots.py
import matplotlib.pyplot as plt


def test_grid_axes_returns_correct_count():
    from profiling.plots import grid_axes
    fig, axes = grid_axes(n_panels=5, ncols=2)
    assert len(axes) >= 5
    plt.close("all")


def test_grid_axes_returns_types():
    from profiling.plots import grid_axes
    fig, axes = grid_axes(n_panels=4, ncols=2)
    assert isinstance(fig, plt.Figure)
    assert all(isinstance(ax, plt.Axes) for ax in axes)
    plt.close("all")


def test_grid_axes_single_panel():
    from profiling.plots import grid_axes
    fig, axes = grid_axes(n_panels=1, ncols=1)
    assert len(axes) == 1
    plt.close("all")
