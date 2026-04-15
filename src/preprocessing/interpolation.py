import numpy as np
from scipy.interpolate import interp1d


def lin_interpolation(data, length):
    """Interpolate a measurement array to a fixed number of samples at uniform arc-length spacing.

    Samples are placed at equal intervals of 2D arc length R = sqrt(deltaX^2 + deltaY^2),
    so curved trajectories are sampled uniformly in space rather than uniformly in X.

    data columns: [x, y, (z,) phase]  — z is optional (present when data has 4 columns).
    Returns an array of shape (length, ncols).
    """
    x = data[:, 0]
    y = data[:, 1]

    # Cumulative 2D arc length along the trajectory
    seg_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate(([0.0], np.cumsum(seg_lengths)))

    # Remove duplicate arc-length values (stationary frames); interp1d needs strictly increasing s
    s, unique_idx = np.unique(s, return_index=True)
    data = data[unique_idx]
    x = data[:, 0]
    y = data[:, 1]

    # Uniformly spaced arc-length positions to sample at
    sd = np.linspace(0.0, s[-1], length)

    x_interp = interp1d(s, x, kind='linear', bounds_error=False, fill_value="extrapolate")(sd)
    y_interp = interp1d(s, y, kind='linear', bounds_error=False, fill_value="extrapolate")(sd)

    if data.shape[1] > 3:
        z = [data[0, 2]] * length
        phase = data[:, 3]
        phase_interp = interp1d(s, phase, kind='linear', bounds_error=False,
                                fill_value="extrapolate")(sd)
        return np.vstack((x_interp, y_interp, z, phase_interp)).T
    else:
        phase = data[:, 2]
        phase_interp = interp1d(s, phase, kind='linear', bounds_error=False,
                                fill_value="extrapolate")(sd)
        return np.vstack((x_interp, y_interp, phase_interp)).T
