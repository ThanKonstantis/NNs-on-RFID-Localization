"""Physics-based RFID localization baseline: Phase Relock.

Non-linear least-squares that optimises tag position [x, y, z] and a
per-antenna phase offset simultaneously, using the physical relationship:

    phase_measured = (4π / λ) * dist(tag, antenna) + phase_offset

Supports 1, 2, 3, and 4 antennas.  Bounds and initial guess are taken directly
from the original notebook experiments for each antenna count.
Note: 1-antenna SAR localization only works for curved trajectories (S, V path)
where the antenna's movement provides sufficient geometric diversity.
"""

import numpy as np
from scipy.optimize import least_squares

from src.evaluation.metrics import _plot_3d

# ── Physical constants ────────────────────────────────────────────────────────
FREQ   = 866e6          # RFID carrier frequency (Hz)
C      = 299792458.0    # speed of light (m/s)
LAMBDA = C / FREQ       # wavelength (m)
CM2M   = 1.0 / 100.0   # raw data is in cm; optimizer works in metres

# ── Per-antenna-count optimisation config ────────────────────────────────────
# Position bounds for [x_tag, y_tag, z_tag] in metres
_XYZ_BOUNDS = {
    1: ([0.5, 0.5, 0.0], [2.4, 1.0, 2.0]),
    2: ([0.5, 0.5, 0.0], [2.4, 1.0, 2.0]),
    3: ([0.5, 0.5, 0.0], [2.4, 1.0, 2.0]),
    4: ([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]),
}
_XYZ_START = {
    1: [1.43, 0.82, 0.5],
    2: [1.43, 0.82, 0.5],
    3: [1.43, 0.82, 0.5],
    4: [1.00, 0.70, 1.0],
}


def _residuals(params, antennas, phases):
    """Objective function for scipy.optimize.least_squares.

    Returns a flat array of raw residuals (one per antenna × timestep).
    least_squares minimises 0.5 * ||residuals||², i.e. the standard sum of
    squared residuals across all antennas and all timesteps.

    params:   [x_tag, y_tag, z_tag, offset_0, ..., offset_{n-1}]
    antennas: list of (x_arr, y_arr, z_arr) tuples in metres, one per antenna
    phases:   list of phase arrays (radians), one per antenna
    """
    x_t, y_t, z_t = params[:3]
    parts = []
    for k, ((xa, ya, za), ph) in enumerate(zip(antennas, phases)):
        offset = params[3 + k]
        dist = np.sqrt((xa - x_t) ** 2 + (ya - y_t) ** 2 + (za - z_t) ** 2)
        parts.append(ph - offset - (4.0 * np.pi / LAMBDA) * dist)
    return np.concatenate(parts)


def _fit_one(sample_cm, n_antennas):
    """Fit the tag position for one sample.

    sample_cm: (seq_len, 4*n_antennas)  raw data in cm
               columns per antenna: [x, y, z, phase]
    Returns estimated tag position in metres.
    """
    antennas, phases = [], []
    for k in range(n_antennas):
        col = k * 4
        xa, ya, za = (sample_cm[:, col:col + 3] * CM2M).T
        antennas.append((xa, ya, za))
        phases.append(sample_cm[:, col + 3])

    phase_min = min(ph.min() for ph in phases)

    lb_xyz, ub_xyz = _XYZ_BOUNDS[n_antennas]
    start = _XYZ_START[n_antennas] + [phase_min] * n_antennas
    lb    = lb_xyz + [-1000.0] * n_antennas
    ub    = ub_xyz + [1000.0]  * n_antennas

    res = least_squares(
        _residuals,
        x0=np.asarray(start, dtype=float),
        args=(antennas, phases),
        method="trf",
        bounds=(lb, ub),
    )
    return res.x[:3]   # [x_tag, y_tag, z_tag] in metres


def run_phase_relock(X_raw, y_true_cm, n_antennas, save_path=None):
    """Run the Phase Relock baseline on a holdout set.

    Parameters
    ----------
    X_raw:      (N, seq_len, 4*n_antennas)  raw antenna data in cm
    y_true_cm:  (N, 3)                      ground-truth tag positions in cm
    n_antennas: 2, 3, or 4
    save_path:  optional path for 3-D prediction scatter PNG

    Returns
    -------
    dict with keys: model_name, mean_distance_error_cm, std, distances
    (same schema as eval_model_3d)
    """
    pred_list, errors = [], []

    for i in range(len(X_raw)):
        pred_m = _fit_one(X_raw[i], n_antennas)
        pred_list.append(pred_m)
        gt_m = y_true_cm[i] * CM2M
        errors.append(np.linalg.norm(gt_m - pred_m) * 100.0)  # m → cm

    pred_cm   = np.array(pred_list) * 100.0  # m → cm
    distances = np.array(errors)
    mean_err  = float(np.mean(distances))
    std_err   = float(np.std(distances))

    if save_path:
        _plot_3d(y_true_cm, pred_cm, distances, mean_err, std_err, save_path=save_path)

    return {
        "model_name":             "PhaseRelock",
        "mean_distance_error_cm": mean_err,
        "std":                    std_err,
        "distances":              distances,
    }
