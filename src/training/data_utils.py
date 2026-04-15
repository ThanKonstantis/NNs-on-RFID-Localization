"""Data loading and normalisation utilities.

All functions are parameterised by `n_antennas` so the same code works for
2-, 3-, and 4-antenna experiments without duplication.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def _normalise(X_train_raw, y_train, X_test_raw, y_test, n_antennas):
    """In-place z-offset removal + abs-max normalisation.

    Modifies the input arrays and returns normalised copies plus the Y scaler.
    """
    # Z-column indices: every 4th column starting at 2 (X, Y, Z, phase per antenna)
    z_indices = [2 + 4 * i for i in range(n_antennas)]

    for dataset, labels in [(X_train_raw, y_train), (X_test_raw, y_test)]:
        for i in range(len(dataset)):
            minimum = np.min([dataset[i, :, z_idx] for z_idx in z_indices])
            for z_idx in z_indices:
                dataset[i, :, z_idx] -= minimum
            labels[i, 2] -= minimum
            # Remove phase offset (every 4th column: 3, 7, 11, …)
            for j in range(dataset.shape[2]):
                if (j + 1) % 4 == 0:
                    dataset[i, :, j] -= np.min(dataset[i, :, j])

    # Per-feature abs-max normalisation (shared across antennas for spatial dims)
    abs_max = np.abs(X_train_raw).max(axis=(0, 1))
    abs_max = abs_max.reshape(n_antennas, 4)
    col_max = abs_max.max(axis=0)
    # Force X, Y, Z to share the same scale
    shared = max(col_max[0], col_max[1], col_max[2])
    col_max[0] = col_max[1] = col_max[2] = shared
    abs_max = np.tile(col_max, (n_antennas, 1)).flatten()
    abs_max[abs_max == 0] = 1.0

    X_train_norm = X_train_raw / abs_max
    X_test_norm = X_test_raw / abs_max

    X_train_scaled = X_train_norm.reshape(len(X_train_norm), -1)
    X_test_scaled = X_test_norm.reshape(len(X_test_norm), -1)

    scaler_Y = StandardScaler()
    y_train_scaled = scaler_Y.fit_transform(y_train)
    y_test_scaled = scaler_Y.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_Y


def make_dataloaders(X_train_raw, y_train, X_test_raw, y_test,
                     n_antennas: int, batch_size: int = 32, seed: int = 42):
    """Normalise data and return (train_loader, test_loader, scaler_Y)."""
    X_tr, X_te, y_tr, y_te, scaler = _normalise(
        X_train_raw.copy(), y_train.copy(),
        X_test_raw.copy(), y_test.copy(),
        n_antennas,
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)

    return train_loader, test_loader, scaler


def make_rnn_dataloaders(X_train_raw, y_train, X_test_raw, y_test,
                         n_antennas: int, batch_size: int = 32, seed: int = 42):
    """Like make_dataloaders but keeps the time-series shape for RNN input.

    Output tensor shape: (N, seq_len, features) as expected by nn.LSTM with batch_first=True.
    """
    X_tr, X_te, y_tr, y_te, scaler = _normalise(
        X_train_raw.copy(), y_train.copy(),
        X_test_raw.copy(), y_test.copy(),
        n_antennas,
    )

    seq_len = X_train_raw.shape[1]
    n_features = n_antennas * 4

    X_tr_t = torch.tensor(X_tr.reshape(-1, seq_len, n_features), dtype=torch.float32)
    X_te_t = torch.tensor(X_te.reshape(-1, seq_len, n_features), dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)

    return train_loader, test_loader, scaler


def make_cnn_dataloaders(X_train_raw, y_train, X_test_raw, y_test,
                         n_antennas: int, batch_size: int = 32, seed: int = 42):
    """Like make_dataloaders but keeps the time-series shape for CNN input (no flatten)."""
    X_tr, X_te, y_tr, y_te, scaler = _normalise(
        X_train_raw.copy(), y_train.copy(),
        X_test_raw.copy(), y_test.copy(),
        n_antennas,
    )

    # Reshape back to (N, features, timesteps) for Conv1d
    seq_len = X_train_raw.shape[1]
    n_features = n_antennas * 4

    X_tr_t = torch.tensor(X_tr.reshape(-1, seq_len, n_features), dtype=torch.float32).permute(0, 2, 1)
    X_te_t = torch.tensor(X_te.reshape(-1, seq_len, n_features), dtype=torch.float32).permute(0, 2, 1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)

    return train_loader, test_loader, scaler
