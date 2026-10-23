from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


# ------------------------------
# Global settings / outputs
# ------------------------------
N_ITER = 1000
N_STIM = 1000
N_REP = 10
N_FEATURES = 20
N_TRAIN_VALUES = [25, 40, 80, 160, 640, 2560]
SNR_VALUES = [0.025, 0.1, 0.3]
RNG_SEED = 0

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figure"
RESULTS_DIR = ROOT / "sim"
FIG_PATH = FIG_DIR / "ols_learning_curve.pdf"
RESULTS_PATH = RESULTS_DIR / "ols_results.csv"
SUMMARY_PATH = RESULTS_DIR / "ols_summary.csv"


# ------------------------------
# Helpers
# ------------------------------
def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def spearman_brown(r):
    return (2 * r) / (1 + r)


def fit_ols(x, y):
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    weights = np.linalg.lstsq(x - x_mean, y - y_mean, rcond=None)[0]
    intercept = y_mean - x_mean @ weights
    return weights, intercept


# ------------------------------
# Main simulation
# ------------------------------
def run_case(snr, case_idx):
    rng = np.random.default_rng(RNG_SEED + case_idx * 100_000)
    weights_true = np.ones(N_FEATURES) / np.sqrt(N_FEATURES)
    noise_sd = np.sqrt(1.0 / snr)
    reliability = 1.0 / (1.0 + noise_sd ** 2 / N_REP)
    ceiling = np.sqrt(reliability)
    results = []

    for iteration in range(N_ITER):
        # Independent test stimuli with repeated noisy responses.
        x_test = rng.standard_normal((N_STIM, N_FEATURES))
        signal_test = x_test @ weights_true
        reps_test = signal_test[:, None] + noise_sd * rng.standard_normal((N_STIM, N_REP))
        y_test = reps_test.mean(axis=1)

        half = N_REP // 2
        split_r = corr(reps_test[:, :half].mean(axis=1),
                       reps_test[:, half:].mean(axis=1))
        sb_est = spearman_brown(split_r)
        oracle_r = corr(signal_test, y_test)

        # Training responses are means of N_REP independent repetitions.
        x_train = rng.standard_normal((max(N_TRAIN_VALUES), N_FEATURES))
        signal_train = x_train @ weights_true
        reps_train = signal_train[:, None] + noise_sd * rng.standard_normal(
            (max(N_TRAIN_VALUES), N_REP)
        )
        y_train = reps_train.mean(axis=1)

        # Training sets are nested; all sizes use the same held-out test set.
        for n_train in N_TRAIN_VALUES:
            weights, intercept = fit_ols(x_train[:n_train], y_train[:n_train])
            prediction = x_test @ weights + intercept
            model_r = corr(prediction, y_test)

            # Population correlation conditional on the fitted weights.
            population_r = ceiling * (weights_true @ weights) / np.linalg.norm(weights)

            results.append({
                "snr": snr,
                "iteration": iteration,
                "n_train": n_train,
                "n_test": N_STIM,
                "n_rep": N_REP,
                "n_features": N_FEATURES,
                "seed": RNG_SEED + case_idx * 100_000,
                "noise_sd": noise_sd,
                "reliability": reliability,
                "ceiling": ceiling,
                "split_r": split_r,
                "sb_est": sb_est,
                "oracle_r": oracle_r,
                "model_r": model_r,
                "model_r2": model_r ** 2,
                "population_r": population_r,
                "model_minus_sb": model_r - sb_est,
                "model_minus_reliability": model_r - reliability,
            })

    return pd.DataFrame(results)


def summarize(df):
    metrics = ["sb_est", "oracle_r", "model_r", "model_r2", "population_r",
               "model_minus_sb", "model_minus_reliability"]
    rows = []
    for (snr, n_train), group in df.groupby(["snr", "n_train"]):
        row = {
            "snr": snr,
            "n_train": n_train,
            "n_iter": len(group),
            "reliability": group["reliability"].iloc[0],
            "ceiling": group["ceiling"].iloc[0],
            "n_nonpositive_sb": int((group["sb_est"] <= 0).sum()),
        }
        for metric in metrics:
            values = group[metric].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {metric}: SNR={snr}, n_train={n_train}")
            mean = values.mean()
            sem = values.std(ddof=1) / np.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sem"] = sem
            row[f"{metric}_ci_low"] = mean - 1.96 * sem
            row[f"{metric}_ci_high"] = mean + 1.96 * sem
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------
# Plotting
# ------------------------------
def plot_learning_curve(summary, fig_path):
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(SNR_VALUES), figsize=(11.5, 4.2), sharey=True,
                             squeeze=False)
    colors = {"model": "#3C5CFF", "reliability": "#B44C43", "ceiling": "#34865A"}

    for ax, (snr, group) in zip(axes[0], summary.groupby("snr")):
        group = group.sort_values("n_train")
        n_train = group["n_train"].to_numpy()
        ax.plot(n_train, group["model_r_mean"], "o-", color=colors["model"], lw=2, ms=4)
        ax.fill_between(n_train, group["model_r_ci_low"], group["model_r_ci_high"],
                        color=colors["model"], alpha=0.18, linewidth=0)
        ax.plot(n_train, group["sb_est_mean"], "--", color=colors["reliability"], lw=2)
        ax.fill_between(n_train, group["sb_est_ci_low"], group["sb_est_ci_high"],
                        color=colors["reliability"], alpha=0.12, linewidth=0)
        ax.axhline(group["ceiling"].iloc[0], color=colors["ceiling"], ls=":", lw=2)
        ax.set_xscale("log")
        ax.set_xticks([25, 80, 640, 2560])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_xlabel("Training stimuli")
        ax.set_title(f"Test reliability = {group['reliability'].iloc[0]:.2f}")
        ax.tick_params(direction="out")

    axes[0, 0].set_ylabel("Pearson correlation / reliability")
    handles = [
        Line2D([], [], color=colors["model"], marker="o", lw=2, label="Fitted-model correlation"),
        Line2D([], [], color=colors["reliability"], ls="--", lw=2, label="Estimated test reliability"),
        Line2D([], [], color=colors["ceiling"], ls=":", lw=2, label="Population correlation ceiling"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.025, "Shading: 95% Monte Carlo confidence intervals for the means.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------
# Main
# ------------------------------
def main():
    if N_REP < 2 or N_REP % 2:
        raise ValueError("N_REP must be positive and even.")
    if N_ITER < 2 or min(N_TRAIN_VALUES) <= N_FEATURES + 1:
        raise ValueError("Use at least two iterations and more training stimuli than fitted parameters.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for case_idx, snr in enumerate(SNR_VALUES):
        print(f"Running OLS simulation (SNR {snr:g})...", flush=True)
        frames.append(run_case(snr, case_idx))

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False)
    summary = summarize(results)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_learning_curve(summary, FIG_PATH)
    print(f"Saved {RESULTS_PATH}")
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
