"""Standard plots produced for every model run.

All plots accept a destination path and just save without showing —
matplotlib's ``Agg`` backend is forced in the trainer module so tests
work headless. The figures are intentionally compact so the experiment
folder stays light.
"""

from __future__ import annotations

from pathlib import Path

# Do NOT force matplotlib.use("Agg") at module import — that would propagate
# to anyone that imports this module (notebooks, downstream scripts) and
# break interactive figures. Tests opt into Agg via their own conftest.py.
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_loss_curve(history: list[dict], out_path: str | Path) -> Path:
    out_path = _ensure_dir(out_path)
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", float("nan")) for h in history]
    val_corr = [h.get("val_weighted_pearson", float("nan")) for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, color="tab:blue", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_corr, color="tab:orange", marker="o", label="val weighted Pearson")
    ax2.set_ylabel("val weighted Pearson", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=0.7)

    fig.suptitle("Training history")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_pred_vs_target_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    *,
    target_names: tuple[str, ...] = ("t0", "t1"),
    max_points: int = 50_000,
) -> Path:
    out_path = _ensure_dir(out_path)
    n_targets = y_true.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4), squeeze=False)
    rng = np.random.default_rng(0)
    if y_true.shape[0] > max_points:
        idx = rng.choice(y_true.shape[0], size=max_points, replace=False)
    else:
        idx = np.arange(y_true.shape[0])
    for i, name in enumerate(target_names[:n_targets]):
        ax = axes[0, i]
        ax.scatter(
            y_true[idx, i], y_pred[idx, i],
            alpha=0.15, s=4, color="tab:blue",
        )
        lim = max(abs(y_true[idx, i]).max(), abs(y_pred[idx, i]).max(), 1.0)
        ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{name} (true)")
        ax.set_ylabel(f"{name} (pred)")
        ax.set_title(name)
        ax.grid(alpha=0.2)
    fig.suptitle("Predictions vs. ground truth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_per_sequence_corr_hist(
    per_seq_corr: dict[int, float],
    out_path: str | Path,
) -> Path:
    out_path = _ensure_dir(out_path)
    vals = np.fromiter(per_seq_corr.values(), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=50, color="tab:orange", edgecolor="white")
    ax.axvline(np.mean(vals), color="black", linestyle="--",
               label=f"mean {np.mean(vals):+.3f}")
    ax.axvline(np.median(vals), color="gray", linestyle=":",
               label=f"median {np.median(vals):+.3f}")
    ax.set_xlabel("weighted Pearson per seq_ix")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Distribution of per-sequence correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
