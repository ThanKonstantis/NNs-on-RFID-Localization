"""2-D Phase Relock baseline for single-antenna RFID localization.

Physical model (per timestep):
    phase = phase_offset + (4π / λ) * sqrt((x_tag - x_ant)² + (y_tag - y_ant)²)

Residuals are raw (unsquared) so that scipy.optimize.least_squares minimises
the standard sum of squared residuals: 0.5 * Σ (phase - offset - (4π/λ)*dist)².
"""

import numpy as np
from scipy.optimize import least_squares

# ── Physical constants (matches notebook) ────────────────────────────────────
FREQ   = 866e6          # RFID carrier frequency (Hz)
C      = 299792458.0    # speed of light (m/s)
LAMBDA = C / FREQ       # wavelength (m)
CM2M   = 1.0 / 100.0   # raw data in cm; optimiser works in metres


def _theoretical_2d(params, x_antenna, y_antenna, phase):
    """Raw residuals for least_squares (one per timestep).

    params = [x_tag (m), y_tag (m), phase_offset (rad)]
    x_antenna, y_antenna: antenna trajectory arrays (m)
    phase: measured phase array (rad)
    """
    x_t, y_t, offset = params
    dist = np.sqrt((x_antenna - x_t) ** 2 + (y_antenna - y_t) ** 2)
    return phase - offset - (4.0 * np.pi / LAMBDA) * dist


def _nonlinear_fit_2d(x_ant, y_ant, phase, start=(1.0, 0.5, -50.0)):
    """Fit one tag position using nonlinear least squares (matches notebook).

    Returns result.x = [x_tag, y_tag, phase_offset] in metres.
    """
    res = least_squares(
        _theoretical_2d,
        x0=np.asarray(start, dtype=float),
        args=(x_ant, y_ant, phase),
        method="trf",
    )
    return res.x


def run_phase_relock_2d(X_raw, y_true_cm, save_path=None):
    """Run the 2-D Phase Relock baseline on a holdout set.

    Parameters
    ----------
    X_raw:      (N, seq_len, >=3)  raw data in cm; cols [x_ant, y_ant, phase, ...]
    y_true_cm:  (N, >=2)           ground-truth tag positions in cm [x, y]
    save_path:  optional path for 2-D prediction scatter PNG/PDF

    Returns
    -------
    dict with model_name, mean_distance_error_cm, std, distances
    (same schema as eval_model_2d)
    """
    preds, errors = [], []

    for i in range(len(X_raw)):
        sample  = X_raw[i]
        x_ant   = sample[:, 0] * CM2M   # cm → m
        y_ant   = sample[:, 1] * CM2M
        phase   = sample[:, 2]

        fit = _nonlinear_fit_2d(x_ant, y_ant, phase)   # [x_tag, y_tag, offset] (m)

        pred_cm = fit[:2] * 100.0                       # m → cm
        preds.append(pred_cm)

        gt_m = y_true_cm[i, :2] * CM2M
        errors.append(np.linalg.norm(gt_m - fit[:2]) * 100.0)

    preds     = np.array(preds)
    distances = np.array(errors)
    mean_err  = float(np.mean(distances))
    std_err   = float(np.std(distances))

    if save_path:
        _plot_relock_2d(y_true_cm[:, :2], preds, distances, mean_err, std_err,
                        save_path=save_path)

    return {
        "model_name":             "PhaseRelock2D",
        "mean_distance_error_cm": mean_err,
        "std":                    std_err,
        "distances":              distances,
    }


def _plot_relock_2d(y_true_cm, preds_cm, distances, mean_err, std_err,
                    n=20, save_path=None):
    import matplotlib.pyplot as plt

    n = min(n, len(y_true_cm))
    plt.figure(figsize=(8, 8))

    plt.scatter(y_true_cm[:n, 0], y_true_cm[:n, 1],
                color="blue", label="Ground Truth", s=100)
    plt.scatter(preds_cm[:n, 0], preds_cm[:n, 1],
                color="red",  label="Predicted",    s=100)

    for i in range(n):
        gt, pred = y_true_cm[i], preds_cm[i]
        plt.plot([gt[0], pred[0]], [gt[1], pred[1]],
                 color="gray", linestyle="--", linewidth=1)
        plt.text((gt[0] + pred[0]) / 2, (gt[1] + pred[1]) / 2,
                 f"{distances[i]:.2f} cm",
                 fontsize=9, ha="center", va="bottom", color="black",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.3))

    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.title(
        f"Phase Relock 2D — Ground Truth vs Predicted\n"
        f"Mean: {mean_err:.2f} cm  Std: {std_err:.2f} cm"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        from pathlib import Path
        p = Path(save_path)
        fmt = "pdf" if p.suffix == ".pdf" else "png"
        plt.savefig(p, format=fmt, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
