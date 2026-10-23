"""Lightweight plotting helpers for consistent styling."""
from __future__ import annotations

from typing import Iterable, Dict

import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_PALETTE = "colorblind"
DEFAULT_FONT = "Arial"

_BASE_RC = {
    "axes.edgecolor": "black",
    "axes.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.titlepad": 8,
    "axes.labelpad": 6,
    "axes.grid": False,
    "font.family": "sans-serif",
    "font.sans-serif": [DEFAULT_FONT, "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": True,
}


def set_style(*, font_scale: float = 1.0, palette: str = DEFAULT_PALETTE) -> None:
    """Apply a compact publication-style theme."""
    sns.set_theme(style="whitegrid", context="talk", font_scale=font_scale, palette=palette)
    plt.rcParams.update(_BASE_RC)


def get_palette(labels: Iterable[str], palette: str = DEFAULT_PALETTE) -> Dict[str, tuple]:
    """Return a consistent color mapping for the provided labels."""
    labels = sorted(labels)
    colors = sns.color_palette(palette, n_colors=len(labels))
    return {label: colors[idx] for idx, label in enumerate(labels)}


def figure_grid(n_rows: int, n_cols: int, *, figsize=(12, 6)):
    """Create a matplotlib figure with the configured style."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    return fig, axes
