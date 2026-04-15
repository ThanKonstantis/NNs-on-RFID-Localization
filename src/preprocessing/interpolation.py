import numpy as np
from scipy.interpolate import interp1d


def lin_interpolation(data, length):
    """Linearly interpolate a measurement array to a fixed number of samples.

    data columns: [x, y, (z,) phase]  — z is optional (present when data has 4 columns).
    Returns an array of shape (length, ncols).
    """
    x = data[:, 0]
    y = data[:, 1]

    xd = np.linspace(x[0], x[-1], length)
    y_interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")(xd)

    if data.shape[1] > 3:
        z = [data[0, 2]] * length
        phase = data[:, 3]
        phase_interp = interp1d(x, phase, kind='linear', bounds_error=False,
                                fill_value="extrapolate")(xd)
        return np.vstack((xd, y_interp, z, phase_interp)).T
    else:
        phase = data[:, 2]
        phase_interp = interp1d(x, phase, kind='linear', bounds_error=False,
                                fill_value="extrapolate")(xd)
        return np.vstack((xd, y_interp, phase_interp)).T
