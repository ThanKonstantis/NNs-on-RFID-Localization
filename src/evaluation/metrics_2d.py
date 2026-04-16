"""Evaluation utilities for 2-D localization models.

eval_model_2d matches the notebook's eval_model() exactly:
  - Inverse transform: y_real = y_scaled * scaler[:2]
  - IEEE-style 2-D scatter plot (10 points, dashed error lines, distance annotations)
  - Saves to PDF by default (matches notebook's tag_prediction_plot.pdf)
"""

import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def eval_model_2d(model, data_loader, scaler, device="cpu",
                  verbose=True, save_path=None):
    """Evaluate a 2-D localization model.

    Parameters
    ----------
    model:       trained PyTorch model
    data_loader: DataLoader over the holdout set
    scaler:      abs_max array from data_utils_2d (shape >= 2); only [:2] used
    save_path:   path for prediction scatter plot (PDF or PNG)

    Returns
    -------
    dict with model_name, mean_distance_error_cm, std, distances
    """
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            all_preds.append(model(X).cpu())
            all_targets.append(y.cpu())
    elapsed = time.perf_counter() - t0

    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Inverse normalisation (matches notebook: y_real = y_scaled * scaler[:2])
    y_pred_real = all_preds   * scaler[:2]
    y_true_real = all_targets * scaler[:2]

    distances = np.linalg.norm(y_pred_real - y_true_real, axis=1)
    mean_err  = float(np.mean(distances))
    std_err   = float(np.std(distances))

    if verbose:
        print(f"Evaluation time: {elapsed:.4f} s  |  {len(distances)} samples")
        print(f"Mean error: {mean_err:.2f} cm  Std: {std_err:.2f} cm")

    if verbose or save_path:
        _plot_2d(y_true_real, y_pred_real, distances, mean_err, std_err,
                 save_path=save_path)

    return {
        "model_name":             model.__class__.__name__,
        "mean_distance_error_cm": mean_err,
        "std":                    std_err,
        "distances":              distances,
    }


def _plot_2d(y_true, y_pred, distances, mean_err, std_err, n=10, save_path=None):
    """IEEE-style 2-D scatter plot matching the notebook's figure."""
    matplotlib.rcParams.update({
        "font.family":     "serif",
        "font.size":       8,
        "figure.figsize":  (3.5, 3.5),
        "axes.labelsize":  8,
        "axes.titlesize":  8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.markersize":3,
        "pdf.fonttype":    42,
    })

    n = min(n, len(y_true))
    fig, ax = plt.subplots()

    ax.scatter(y_true[:n, 0], y_true[:n, 1],
               color="black", label="Ground Truth", s=12, zorder=3)
    ax.scatter(y_pred[:n, 0], y_pred[:n, 1],
               color="gray", label="Predicted", s=12, marker="x", zorder=3)

    for i, (gt, pred) in enumerate(zip(y_true[:n], y_pred[:n])):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]],
                color="0.6", linestyle="--", linewidth=0.5, zorder=2)
        mid_x = (gt[0] + pred[0]) / 2
        mid_y = (gt[1] + pred[1]) / 2
        ax.text(mid_x, mid_y, f"{distances[i]:.1f}",
                fontsize=6, ha="center", va="bottom", color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.2))

    ax.set_aspect("equal")
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")
    ax.set_title(
        f"Predicted vs. Ground Truth Tag Positions\n"
        f"Mean: {mean_err:.2f} cm  Std: {std_err:.2f} cm"
    )
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fmt = "pdf" if save_path.suffix == ".pdf" else "png"
        plt.savefig(save_path, format=fmt, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_results_2d(train_arr, test_arr, save_path=None):
    """Plot normalised train/val loss curves (same as 3-D version)."""
    from src.evaluation.metrics import plot_results
    plot_results(train_arr, test_arr, save_path=save_path)
