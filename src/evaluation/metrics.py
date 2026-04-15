"""Evaluation utilities: distance error computation and plotting."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def eval_model_3d(model, data_loader, scaler, device="cpu", verbose=True, save_path=None):
    """Evaluate a 3-D localization model and return error statistics."""
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            all_preds.append(model(X).cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    y_pred_real = scaler.inverse_transform(all_preds)
    y_true_real = scaler.inverse_transform(all_targets)

    distances = np.linalg.norm(y_pred_real - y_true_real, axis=1)
    mean_err = np.mean(distances)
    std_err = np.std(distances)

    if verbose or save_path:
        _plot_3d(y_true_real, y_pred_real, distances, mean_err, std_err, save_path=save_path)

    return {
        "model_name": model.__class__.__name__,
        "mean_distance_error_cm": float(mean_err),
        "std": float(std_err),
        "distances": distances,
    }


def plot_results(train_arr, test_arr, save_path=None):
    """Plot normalised train/test loss curves."""
    def _norm(a):
        mn, mx = np.min(a), np.max(a)
        return (a - mn) / (mx - mn) if mx > mn else a

    plt.figure(figsize=(8, 5))
    plt.plot(_norm(train_arr), label="Train Loss")
    plt.plot(_norm(test_arr), label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Normalised Loss")
    plt.title("Normalised Training Loss")
    plt.grid()
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_cdf(distances, label="Model", save_path=None):
    """Plot the CDF of distance errors."""
    sorted_d = np.sort(distances)
    cdf = np.arange(len(sorted_d)) / (len(sorted_d) - 1)
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_d, cdf, lw=2, label=label)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Distance Error (cm)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Distance Errors")
    plt.xlim(left=0)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300)
    plt.show()


def _plot_3d(y_true, y_pred, distances, mean_err, std_err, n=30, save_path=None):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*y_true[:n].T, color="blue", label="Ground Truth", s=100)
    ax.scatter(*y_pred[:n].T, color="red", label="Predicted", s=100)
    for gt, pred, d in zip(y_true[:n], y_pred[:n], distances[:n]):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], [gt[2], pred[2]],
                color="gray", linestyle="--", linewidth=1)
        mid = (gt + pred) / 2
        ax.text(*mid, f"{d:.1f} cm", fontsize=8, ha="center",
                bbox=dict(facecolor="white", alpha=0.3, edgecolor="none"))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Ground Truth vs Predicted\nMean Error: {mean_err:.2f} cm  Std: {std_err:.2f} cm")
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
