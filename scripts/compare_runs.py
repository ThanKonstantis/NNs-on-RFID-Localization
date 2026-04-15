"""Generate comparison plots for a completed run directory.

Produces:
  <results_dir>/comparison_cdf.png   — all model CDFs on one chart
  <results_dir>/summary_bar.png      — mean error ± std bar chart

Usage:
    python scripts/compare_runs.py --results-dir results/my_run
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run_data(results_dir: Path) -> dict[str, dict]:
    """Return {model_name: {"distances": ndarray, "metrics": dict}} for every
    successful model sub-directory found inside results_dir."""
    data = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        dist_path = model_dir / "metrics" / "distances.npy"
        json_path = model_dir / "metrics" / "metrics.json"
        if not dist_path.exists():
            continue
        distances = np.load(dist_path)
        metrics = {}
        if json_path.exists():
            with open(json_path) as f:
                metrics = json.load(f)
        data[model_dir.name] = {"distances": distances, "metrics": metrics}
    return data


def plot_comparison_cdf(data: dict[str, dict], save_path: Path) -> None:
    """Plot all model CDFs on a single chart, sorted by median error."""
    # Sort models by median distance error (best first)
    order = sorted(data, key=lambda m: np.median(data[m]["distances"]))

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")

    for i, model in enumerate(order):
        distances = np.sort(data[model]["distances"])
        cdf = np.arange(len(distances)) / max(len(distances) - 1, 1)
        mean_err = data[model]["metrics"].get(
            "mean_distance_error_cm", np.mean(distances)
        )
        label = f"{model}  (mean={mean_err:.1f} cm)"
        ax.plot(distances, cdf, lw=1.8, label=label, color=cmap(i % 10))

    ax.set_xlabel("Distance Error (cm)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("CDF of Distance Errors — All Models", fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_summary_bar(data: dict[str, dict], save_path: Path) -> None:
    """Bar chart of mean errors with std error bars, sorted best → worst."""
    order = sorted(
        data,
        key=lambda m: data[m]["metrics"].get(
            "mean_distance_error_cm", np.mean(data[m]["distances"])
        ),
    )

    means = [
        data[m]["metrics"].get("mean_distance_error_cm", np.mean(data[m]["distances"]))
        for m in order
    ]
    stds = [
        data[m]["metrics"].get("std_cm", np.std(data[m]["distances"]))
        for m in order
    ]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 6))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue",
                  error_kw={"elinewidth": 1.2, "ecolor": "black"})

    # Annotate each bar with the mean value
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Distance Error (cm)", fontsize=11)
    ax.set_title("Mean Error ± Std per Model", fontsize=13)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots for a run.")
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to the run directory (e.g. results/my_run)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: '{results_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    data = load_run_data(results_dir)
    if not data:
        print("No model results found — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(data)} model(s): {', '.join(sorted(data))}")

    images_dir = results_dir
    images_dir.mkdir(exist_ok=True)

    plot_comparison_cdf(data, images_dir / "comparison_cdf.png")
    plot_summary_bar(data, images_dir / "summary_bar.png")


if __name__ == "__main__":
    main()
