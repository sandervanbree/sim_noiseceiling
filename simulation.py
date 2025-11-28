import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from plotting import set_style, get_palette

# ------------------------------
# Global settings / outputs
# ------------------------------
N_ITER = 200
N_STIM = 1000
N_REP = 10
RNG_SEED = 0
MODEL_NOISE_SCALE = 0.0
MODEL_NOISE_SCALES = [0.0]

FIG_DIR = "figure"
RESULTS_PATH = Path("sim/results.csv")
FIG_SB_VS_FULL = f"{FIG_DIR}/sb_vs_full.pdf"
FIG_OVERVIEW = f"{FIG_DIR}/overview.pdf"
FIG_CONCEPTUAL_FULL = f"{FIG_DIR}/conceptual_ceiling_full.pdf"
FIG_CONCEPTUAL_BAR = f"{FIG_DIR}/conceptual_bar.pdf"

# SNR range: logarithmic spacing (powers of 2)
SNR_VALUES = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56]
CASES = [dict(label=f"SNR {snr:.2f}", snr=snr) for snr in SNR_VALUES]


# ------------------------------
# Helpers
# ------------------------------
def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def spearman_brown(r):
    r = np.clip(r, -0.999999, 0.999999)
    return (2 * r) / (1 + r)


# ------------------------------
# Main simulation
# ------------------------------
def simulate(noise_sd, model_noise_sd, signal, rng):
    results = []
    for _ in range(N_ITER):
        # Dataset A: signal + noise across reps
        reps_a = signal[:, None] + noise_sd * rng.standard_normal((N_STIM, N_REP))
        dataset_a = reps_a.mean(axis=1)

        # Dataset B: independent noise
        reps_b = signal[:, None] + noise_sd * rng.standard_normal((N_STIM, N_REP))
        dataset_b = reps_b.mean(axis=1)

        # H1: Split-half reliability
        half = N_REP // 2
        split_a = reps_a[:, :half].mean(axis=1)
        split_b = reps_a[:, half:].mean(axis=1)
        split_r = corr(split_a, split_b)
        if np.isnan(split_r):
            continue
        sb_est = spearman_brown(split_r)

        # H1: Full-vs-full reliability
        full_r = corr(dataset_a, dataset_b)

        # H2: Perfect model correlation & explained variance
        perfect_r = corr(signal, dataset_a)
        perfect_r2 = perfect_r ** 2

        # H2: Noisy model
        model = signal + model_noise_sd * rng.standard_normal(N_STIM)
        model_r = corr(model, dataset_a)
        model_r2 = model_r ** 2

        # Derived metrics for plotting compatibility
        sb_pos = sb_est if sb_est > 0 else np.nan
        sqrt_sb = np.sqrt(sb_pos) if not np.isnan(sb_pos) else np.nan
        sb_sq = sb_est ** 2

        results.append({
            'sb_est': sb_est,
            'split_r': split_r,
            'full_r': full_r,
            'sqrt_sb': sqrt_sb,
            'sb_sq': sb_sq,
            'perfect_r': perfect_r,
            'perfect_r2': perfect_r2,
            'model_r': model_r,
            'model_r2': model_r2,
        })

    return pd.DataFrame(results)


def run_case(cfg, case_idx):
    snr = cfg["snr"]
    noise_sd = (1.0 / snr) ** 0.5
    model_noise_sd = MODEL_NOISE_SCALE * noise_sd

    rng = np.random.default_rng(RNG_SEED + case_idx * 100_000 + int(MODEL_NOISE_SCALE * 1000))
    signal = rng.standard_normal(N_STIM)

    frame = simulate(noise_sd, model_noise_sd, signal, rng)

    frame.insert(0, "model_noise_sd", model_noise_sd)
    frame.insert(0, "noise_sd", noise_sd)
    frame.insert(0, "snr", snr)
    frame.insert(0, "case", cfg["label"])
    return frame


def _add_ellipse(ax, mean_x, mean_y, cov, color):
    if np.isnan(mean_x) or np.isnan(mean_y) or np.isnan(cov).any():
        return
    try:
        vals, vecs = np.linalg.eigh(cov)
        eps = 1e-8
        vals = np.maximum(vals, eps)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals)
        ell = Ellipse((mean_x, mean_y), width, height, angle=angle,
                      facecolor=color, edgecolor=color, alpha=0.25, lw=0)
        ax.add_patch(ell)
        ax.plot(mean_x, mean_y, marker="o", markersize=6, color=color, lw=0)
    except Exception:
        pass


def plot_canvas(df, fig_path):
    mask = df["sb_est"].notna()
    if not mask.any():
        return
    data = df.loc[mask].copy()
    data["sb_sq"] = data["sb_est"] ** 2
    groups = list(data.groupby("case"))

    # Use colormap for SNR gradient
    import matplotlib.cm as cm
    n_cases = len(data["case"].unique())
    try:
        cmap = cm.colormaps.get_cmap("plasma")
    except AttributeError:
        cmap = cm.get_cmap("plasma")
    colors = [cmap(i / (n_cases - 1)) for i in range(n_cases)]
    palette = {case: colors[i] for i, (case, _) in enumerate(groups)}

    sb_specs = [
        ("sqrt_sb", "Reliability (√Spearman-Brown)"),
        ("sb_est", "Reliability (Spearman-Brown)"),
        ("sb_sq", "Reliability (Spearman-Brown²)"),
    ]
    model_specs = [
        ("model_r", "Model correlation (r)"),
        ("model_r2", "Model explainable variance (R²)"),
    ]
    tick_positions = np.linspace(0, 1, 6)
    tick_formatter = FuncFormatter(
        lambda v, _: "0" if np.isclose(v, 0) else ("1" if np.isclose(v, 1) else f"{v:.1f}".rstrip('0').rstrip('.'))
    )

    # Compact square panels
    panel_size = 3.6
    margin_l, margin_r = 0.22, 0.04
    margin_b, margin_t = 0.20, 0.08
    n_row, n_col = len(model_specs), len(sb_specs)

    fig = plt.figure(figsize=(panel_size * n_col, panel_size * n_row), dpi=300)
    axes = []
    for row in range(n_row):
        axes_row = []
        for col in range(n_col):
            panel_w = (1 - margin_l - margin_r) / n_col
            panel_h = (1 - margin_b - margin_t) / n_row
            left = margin_l + col * (1.0 / n_col)
            bottom = 1.0 - margin_t - (row + 1) * (1.0 / n_row) + margin_b / n_row
            ax = fig.add_axes([left, bottom, panel_w, panel_h])
            axes_row.append(ax)
        axes.append(axes_row)
    axes = np.array(axes)

    for row, (model_col, model_label) in enumerate(model_specs):
        for col, (sb_col, sb_label) in enumerate(sb_specs):
            ax = axes[row, col]
            ax.set_facecolor("white")
            ax.grid(False)
            ref = np.linspace(0, 1, 400)

            # shade model>ceiling
            ax.fill_between(ref, ref, 1, color="#8b4545", alpha=0.15, zorder=0, linewidth=0)

            ax.plot(ref, ref, color="0.2", lw=3.0, label="Theoretical ceiling")
            for case, sub in groups:
                color = palette[case]
                sub_plot = sub.sample(2000, random_state=0) if len(sub) > 2000 else sub
                ax.scatter(sub_plot[sb_col], sub_plot[model_col], s=20, alpha=0.25,
                           color=color, edgecolor="none")
                mean_x = np.nanmean(sub[sb_col])
                mean_y = np.nanmean(sub[model_col])
                cov = np.cov(np.vstack([sub[sb_col], sub[model_col]]), ddof=0)
                _add_ellipse(ax, mean_x, mean_y, cov, color)
                ax.scatter([mean_x], [mean_y], s=110, facecolor=color, edgecolor="black", linewidth=2.0, zorder=5)

            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2.0)
            ax.spines['bottom'].set_linewidth(2.0)
            ax.spines['left'].set_position(('outward', 0))
            ax.spines['bottom'].set_position(('outward', 0))

            ax.tick_params(axis='both', which='major', labelsize=12, length=6.0, width=1.6,
                           direction='out', top=False, right=False, pad=1.5)
            ax.xaxis.set_major_locator(FixedLocator(tick_positions))
            ax.yaxis.set_major_locator(FixedLocator(tick_positions))
            ax.xaxis.set_major_formatter(tick_formatter)
            ax.yaxis.set_major_formatter(tick_formatter)
            ax.set_xlabel(sb_label, fontsize=12, weight="bold", labelpad=2)
            ax.set_ylabel(model_label, fontsize=12, weight="bold", labelpad=2)
            ax.margins(x=0.04, y=0.0)

            per_case = []
            for _, sub in groups:
                delta = np.abs(sub[model_col] - sub[sb_col])
                val = np.nanmean(delta)
                if not np.isnan(val):
                    per_case.append(val)
            mean_delta = np.nanmean(per_case) if per_case else np.nan
            sem_delta = np.nanstd(per_case, ddof=1) / np.sqrt(len(per_case)) if len(per_case) > 1 else np.nan
            if not np.isnan(mean_delta):
                text = f"Absolute error = {mean_delta:.3f}"
                if not np.isnan(sem_delta):
                    text += f"\nSEM = {sem_delta:.3f}"
                ax.text(0.02, 0.95, text,
                        transform=ax.transAxes, ha="left", va="top", fontsize=11, color="0.1", weight="bold")

    scenario_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7,
               markerfacecolor=palette[case], markeredgecolor='black', markeredgewidth=1.0,
               label=case.replace("SNR ", ""))
        for case, _ in groups
    ]
    legend_handles = scenario_handles + [
        Patch(facecolor="#8b4545", edgecolor="none", alpha=0.45, label="model > ceiling"),
    ]
    axes[0, -1].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=10,
        ncol=2,
        borderpad=0.15,
        labelspacing=0.25,
        handlelength=0.8,
        handletextpad=0.4,
        borderaxespad=0.15,
        columnspacing=0.8,
        title="SNR",
        title_fontsize=10,
    )
    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(fig_path.replace('.pdf', '.png'), dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def plot_sb_vs_full_baseline(df, fig_path=FIG_SB_VS_FULL):
    mask = df["sb_est"].notna() & df["full_r"].notna()
    if not mask.any():
        return
    data = df.loc[mask].copy()
    groups = list(data.groupby("case"))

    # Use plasma colormap for SNR gradient
    import matplotlib.cm as cm
    n_cases = len(data["case"].unique())
    try:
        cmap = cm.colormaps.get_cmap("plasma")
    except AttributeError:
        cmap = cm.get_cmap("plasma")
    colors = [cmap(i / (n_cases - 1)) for i in range(n_cases)]
    palette = {case: colors[i] for i, (case, _) in enumerate(groups)}

    tick_positions = np.linspace(0, 1, 6)
    tick_formatter = FuncFormatter(
        lambda v, _: "0" if np.isclose(v, 0) else ("1" if np.isclose(v, 1) else f"{v:.1f}".rstrip('0').rstrip('.'))
    )

    panel_size = 3.6 * 1.2
    margin_l, margin_r = 0.22, 0.04
    margin_b, margin_t = 0.20, 0.08

    fig = plt.figure(figsize=(panel_size, panel_size), dpi=300)
    ax = fig.add_axes([margin_l, margin_b, 1 - margin_l - margin_r, 1 - margin_b - margin_t])
    ax.set_facecolor("white")
    ax.grid(False)

    ref = np.linspace(0, 1, 400)
    ax.plot(ref, ref, color="0.2", lw=3.0, label="y = x")

    for case, sub in groups:
        color = palette[case]
        sub_plot = sub.sample(2000, random_state=0) if len(sub) > 2000 else sub
        ax.scatter(sub_plot["sb_est"], sub_plot["full_r"], s=20, alpha=0.25, color=color, edgecolor="none")
        mean_x = np.nanmean(sub["sb_est"])
        mean_y = np.nanmean(sub["full_r"])
        cov = np.cov(np.vstack([sub["sb_est"], sub["full_r"]]), ddof=0)
        _add_ellipse(ax, mean_x, mean_y, cov, color)
        ax.scatter([mean_x], [mean_y], s=110, facecolor=color, edgecolor="black", linewidth=2.0, zorder=5)

    per_case_mae = []
    for _, sub in groups:
        delta = np.abs(sub["full_r"] - sub["sb_est"])
        per_case_mae.append(np.nanmean(delta))
    if per_case_mae:
        mean_mae = np.nanmean(per_case_mae)
        sem_mae = np.nanstd(per_case_mae, ddof=1) / np.sqrt(len(per_case_mae)) if len(per_case_mae) > 1 else np.nan
        text = f"Absolute error = {mean_mae:.3f}"
        if not np.isnan(sem_mae):
            text += f"\nSEM = {sem_mae:.3f}"
        ax.text(0.02, 0.95, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=11, color="0.1", weight="bold")

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))

    ax.tick_params(axis='both', which='major', labelsize=12, length=6.0, width=1.6,
                   direction='out', top=False, right=False, pad=1.5)
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.yaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.yaxis.set_major_formatter(tick_formatter)
    ax.set_xlabel("Split-half reliability\n(Spearman-Brown)", fontsize=12, weight="bold", labelpad=2)
    ax.set_ylabel("Full-vs-full reliability\n(Dataset A vs B)", fontsize=12, weight="bold", labelpad=2)
    ax.margins(x=0.04, y=0.0)

    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(fig_path.replace('.pdf', '.png'), dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


# ------------------------------
# New exact conceptual figure (Option 2: two full-M datasets)
# ------------------------------
def plot_conceptual_ceiling(fig_path, invert_x=True,
                            target_wrong=0.50, target_true=0.7071,
                            x_max=4.0, n_points=200):
    """
    - Left side is the fixed noisy DATA average.
    - Right side is the 'true model + noise' average.
    - Curve is exact: r = 1 / sqrt((1+v_A)*(1+v_B)),
      where v_A and v_B are post-averaging noise variances.

    By default, chooses v_A so that:
      equal-noise -> r ≈ 0.50, and noiseless-model -> r ≈ 0.707.
    """
    v_A = (1.0 / (target_true ** 2)) - 1.0

    v_B_vals = np.linspace(0.0, x_max, n_points)
    r_vals = 1.0 / np.sqrt((1.0 + v_A) * (1.0 + v_B_vals))

    vB_equal = v_A
    r_equal = 1.0 / (1.0 + v_A)
    vB_true = 0.0
    r_true = 1.0 / np.sqrt(1.0 + v_A)

    panel_size = 3.6 * 1.2
    margin_l, margin_r = 0.22, 0.04
    margin_b, margin_t = 0.20, 0.08

    fig = plt.figure(figsize=(panel_size, panel_size), dpi=300)
    ax = fig.add_axes([margin_l, margin_b, 1 - margin_l - margin_r, 1 - margin_b - margin_t])
    ax.set_facecolor("white")
    ax.grid(False)

    color_curve = 'black'
    color_wrong = '#d62728'
    color_true = '#2ca02c'

    # exact curve
    ax.plot(v_B_vals, r_vals, color=color_curve, lw=3.0, zorder=3)

    # wrong ceiling (equal noise) - horizontal dashed line
    ax.axhline(r_equal, color=color_wrong, lw=2.5, ls='--', alpha=0.7, zorder=2)
    ax.plot(vB_equal, r_equal, 'o', ms=10, color=color_wrong,
            markeredgecolor='black', markeredgewidth=2.0, zorder=4)
    ax.text(x_max*0.05 if invert_x else x_max*0.95, r_equal+0.02,
            'Wrong ceiling', ha='left' if invert_x else 'right', va='bottom',
            fontsize=11, weight='bold', color=color_wrong)

    # true ceiling (noiseless model) - horizontal dashed line
    ax.axhline(r_true, color=color_true, lw=2.5, ls='--', alpha=0.7, zorder=2)
    ax.plot(vB_true, r_true, 's', ms=10, color=color_true,
            markeredgecolor='black', markeredgewidth=2.0, zorder=4)
    ax.text(x_max*0.05 if invert_x else x_max*0.95, r_true+0.02,
            'True ceiling', ha='left' if invert_x else 'right', va='bottom',
            fontsize=11, weight='bold', color=color_true)

    # model performance (blue horizontal line)
    color_model = '#3C5CFF'
    r_model = 0.60
    ax.axhline(r_model, color=color_model, lw=3.0, ls='-', alpha=1.0, zorder=2)
    ax.text(x_max*0.95 if not invert_x else x_max*0.05, r_model-0.01,
            'Model', ha='right' if not invert_x else 'left', va='top',
            color=color_model, fontsize=11, weight='bold')

    # axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))
    ax.tick_params(axis='both', which='major', labelsize=12, length=6.0, width=1.6,
                   direction='out', top=False, right=False, pad=1.5)

    def clean(v, _):
        return f'{int(v)}' if np.isclose(v, round(v)) else f'{v:.2f}'.rstrip('0').rstrip('.')

    ax.xaxis.set_major_formatter(FuncFormatter(clean))
    ax.yaxis.set_major_formatter(FuncFormatter(clean))
    ax.set_xticks(np.arange(0, x_max + 0.5, 0.5 if x_max <= 2 else 1.0))
    ax.set_ylim(0.45, 1.02)

    ax.set_xlabel('$\sigma^2_B$ (True model + noise)', fontsize=12, weight='bold', labelpad=2)
    ax.set_ylabel('Data vs True model + noise (Pearson $r$)', fontsize=12, weight='bold', labelpad=2)

    if invert_x:
        ax.invert_xaxis()

    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(fig_path.replace('.pdf', '.png'), dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


def plot_conceptual_bar(fig_path, target_true=0.7071, r_model=0.60):
    """Simple bar plot showing delta between model and wrong/true ceilings."""
    v_A = (1.0 / (target_true ** 2)) - 1.0
    r_equal = 1.0 / (1.0 + v_A)  # wrong ceiling
    r_true = 1.0 / np.sqrt(1.0 + v_A)  # true ceiling

    delta_to_wrong = r_equal - r_model
    delta_to_true = r_true - r_model

    panel_size = 3.6 * 1.2
    margin_l, margin_r = 0.22, 0.04
    margin_b, margin_t = 0.20, 0.08

    fig = plt.figure(figsize=(panel_size, panel_size), dpi=300)
    ax = fig.add_axes([margin_l, margin_b, 1 - margin_l - margin_r, 1 - margin_b - margin_t])
    ax.set_facecolor("white")
    ax.grid(False)

    color_wrong = '#d62728'
    color_true = '#2ca02c'

    x_positions = [1, 2]
    x_labels = ['Gap to\nwrong ceiling', 'Gap to\ntrue ceiling']
    bar_heights = [delta_to_wrong, delta_to_true]
    colors = [color_wrong, color_true]

    # bars
    bars = ax.bar(x_positions, bar_heights, width=0.6, color=colors,
                  edgecolor='black', linewidth=2.0)

    # value labels on top of bars
    for i, (x, h) in enumerate(zip(x_positions, bar_heights)):
        ax.text(x, h + 0.01, f'{h:.2f}', ha='center', va='bottom',
                fontsize=12, weight='bold', color='black')

    # axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))
    ax.tick_params(axis='y', which='major', labelsize=12, length=6.0, width=1.6,
                   direction='out', left=True, right=False, pad=1.5)
    ax.tick_params(axis='x', which='major', labelsize=11, length=0, width=0, pad=8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, weight='bold')
    ax.set_ylabel('Gap in correlation (Pearson $r$)', fontsize=12, weight='bold', labelpad=2)
    ax.set_ylim(0, max(bar_heights) * 1.15)
    ax.set_xlim(0.4, 2.6)

    def clean(v, _):
        return f'{int(v)}' if np.isclose(v, round(v)) else f'{v:.2f}'.rstrip('0').rstrip('.')
    ax.yaxis.set_major_formatter(FuncFormatter(clean))

    fig.savefig(fig_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(fig_path.replace('.pdf', '.png'), dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


# ------------------------------
# Overview (unchanged)
# ------------------------------
def plot_overview(fig_path=FIG_OVERVIEW):
    rng = np.random.default_rng(RNG_SEED)
    snr = 0.30
    noise_sd = (1.0 / snr) ** 0.5
    signal = rng.standard_normal(N_STIM)
    reps = signal[:, None] + noise_sd * rng.standard_normal((N_STIM, N_REP))
    dataset = reps.mean(axis=1)
    model = signal + 0.3 * noise_sd * rng.standard_normal(N_STIM)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(signal, color="black", lw=2)
    axes[0].set_title("Signal")
    axes[0].set_ylabel("Value")
    axes[0].set_xlabel("Stimulus")
    axes[0].grid(alpha=0.2)

    axes[1].plot(dataset, color="#1f77b4", lw=1.6, label="Mean across reps")
    for i in range(0, N_STIM, N_STIM // 12):
        axes[1].scatter([i] * N_REP, reps[i], s=10, alpha=0.2, color="#1f77b4")
    axes[1].set_title(f"Dataset (SNR = {snr:.2f})")
    axes[1].set_xlabel("Stimulus")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].plot(dataset, color="#1f77b4", lw=1.6, label="Dataset")
    axes[2].plot(model, color="#ff7f0e", lw=1.6, label="Model")
    axes[2].set_title("Model vs Dataset")
    axes[2].set_xlabel("Stimulus")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, transparent=True)
    fig.savefig(fig_path.replace('.pdf', '.png'), dpi=150, transparent=True)
    plt.close(fig)


# ------------------------------
# Main
# ------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    set_style()

    print("Generating overview figure...")
    plot_overview()

    print("Generating conceptual ceiling figure (exact, option 2)...")
    plot_conceptual_ceiling(FIG_CONCEPTUAL_FULL, invert_x=True, target_true=0.8, x_max=2.0)
    print(f"  Saved {FIG_CONCEPTUAL_FULL}")

    print("Generating conceptual bar plot...")
    plot_conceptual_bar(FIG_CONCEPTUAL_BAR, target_true=0.8, r_model=0.60)
    print(f"  Saved {FIG_CONCEPTUAL_BAR}")

    all_frames = []
    for scale in MODEL_NOISE_SCALES:
        global MODEL_NOISE_SCALE
        MODEL_NOISE_SCALE = scale
        print(f"\nRunning simulations (model noise scale {scale})...")

        frames = [run_case(cfg, i) for i, cfg in enumerate(CASES)]
        df = pd.concat(frames, ignore_index=True)
        all_frames.append(df)

        suffix = f"_noise_{int(scale*100):02d}"
        fig_path = f"{FIG_DIR}/canvas{suffix}.pdf"
        plot_canvas(df, fig_path=fig_path)
        print(f"  Saved {fig_path}")

    plot_sb_vs_full_baseline(all_frames[0], fig_path=FIG_SB_VS_FULL)
    print(f"  Saved {FIG_SB_VS_FULL}")

    combined = pd.concat(all_frames, ignore_index=True)
    summary = combined.groupby(["case", "model_noise_sd"], as_index=False).agg({
        "snr": "first",
        "sb_est": "mean",
        "sqrt_sb": "mean",
        "full_r": "mean",
        "perfect_r": "mean",
        "perfect_r2": "mean",
        "model_r": "mean",
        "model_r2": "mean",
    }).round(3)
    summary.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
