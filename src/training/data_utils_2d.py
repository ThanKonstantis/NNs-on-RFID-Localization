"""Data loading and normalisation utilities for 2-D (single-antenna) experiments.

Preprocessing matches the notebook's data_func_cv / data_func_eval exactly:
  - Column 2 of each sample is the phase channel; its per-sample minimum is subtracted.
  - X and Y position columns (0 and 1) share the same abs-max scale.
  - Y labels are normalised by abs_max[:2] (same spatial units as X).
  - Inverse transform: y_real = y_scaled * abs_max[:2]
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ─── Internal normalisation helpers ──────────────────────────────────────────

def _phase_offset_removal(X):
    """Subtract per-sample minimum from column 2 (phase channel). In-place."""
    for i in range(len(X)):
        X[i, :, 2] -= np.min(X[i, :, 2])


def _abs_max_normalise(X_train, X_other):
    """Compute abs-max on training data and normalise both splits.

    Forces columns 0 (X) and 1 (Y) to share the same scale.
    Returns (X_train_norm, X_other_norm, abs_max).
    """
    abs_max = np.abs(X_train).max(axis=(0, 1))   # shape: (n_features,)
    abs_max[1] = abs_max[0]                       # Y shares X scale
    abs_max[abs_max == 0] = 1.0
    return X_train / abs_max, X_other / abs_max, abs_max


# ─── Cross-validation fold dataloaders (flat MLP input) ──────────────────────

def make_cv_dataloaders_2d(X_all, y_all, train_idx, val_idx,
                           batch_size: int = 32, seed: int = 42):
    """Build (train_loader, val_loader, abs_max) for one CV fold.

    Uses fancy-index copies so the original array is never modified.
    y_all must already be sliced to 2 columns [:,  :2].
    """
    X_train = X_all[train_idx].copy()
    X_val   = X_all[val_idx].copy()
    y_train = y_all[train_idx]
    y_val   = y_all[val_idx]

    _phase_offset_removal(X_train)
    _phase_offset_removal(X_val)

    X_tr_norm, X_val_norm, abs_max = _abs_max_normalise(X_train, X_val)

    X_tr_flat  = X_tr_norm.reshape(len(X_tr_norm), -1)
    X_val_flat = X_val_norm.reshape(len(X_val_norm), -1)

    y_tr_scaled  = y_train / abs_max[:2]
    y_val_scaled = y_val   / abs_max[:2]

    return _to_loaders(X_tr_flat, y_tr_scaled, X_val_flat, y_val_scaled,
                       batch_size, seed, abs_max)


# ─── Final evaluation dataloaders (flat MLP input) ───────────────────────────

def make_eval_dataloaders_2d(X_main, X_holdout, y_main, y_holdout,
                             batch_size: int = 32, seed: int = 42):
    """Build (train_loader, test_loader, abs_max) for final holdout evaluation.

    Modifies X_main and X_holdout in-place (matches notebook data_func_eval).
    y_main / y_holdout must already be sliced to 2 columns.
    """
    _phase_offset_removal(X_main)
    _phase_offset_removal(X_holdout)

    X_tr_norm, X_te_norm, abs_max = _abs_max_normalise(X_main, X_holdout)

    X_tr_flat = X_tr_norm.reshape(len(X_tr_norm), -1)
    X_te_flat = X_te_norm.reshape(len(X_te_norm), -1)

    y_tr_scaled = y_main    / abs_max[:2]
    y_te_scaled = y_holdout / abs_max[:2]

    return _to_loaders(X_tr_flat, y_tr_scaled, X_te_flat, y_te_scaled,
                       batch_size, seed, abs_max)


# ─── RNN variants (sequence shape preserved) ─────────────────────────────────

def make_cv_dataloaders_2d_rnn(X_all, y_all, train_idx, val_idx,
                                batch_size: int = 32, seed: int = 42):
    """CV fold dataloaders keeping the (N, seq_len, features) shape for RNNs."""
    X_train = X_all[train_idx].copy()
    X_val   = X_all[val_idx].copy()
    y_train = y_all[train_idx]
    y_val   = y_all[val_idx]

    _phase_offset_removal(X_train)
    _phase_offset_removal(X_val)

    X_tr_norm, X_val_norm, abs_max = _abs_max_normalise(X_train, X_val)

    y_tr_scaled  = y_train / abs_max[:2]
    y_val_scaled = y_val   / abs_max[:2]

    return _to_loaders(X_tr_norm, y_tr_scaled, X_val_norm, y_val_scaled,
                       batch_size, seed, abs_max)


def make_eval_dataloaders_2d_rnn(X_main, X_holdout, y_main, y_holdout,
                                  batch_size: int = 32, seed: int = 42):
    """Final evaluation dataloaders keeping sequence shape for RNNs. In-place."""
    _phase_offset_removal(X_main)
    _phase_offset_removal(X_holdout)

    X_tr_norm, X_te_norm, abs_max = _abs_max_normalise(X_main, X_holdout)

    y_tr_scaled = y_main    / abs_max[:2]
    y_te_scaled = y_holdout / abs_max[:2]

    return _to_loaders(X_tr_norm, y_tr_scaled, X_te_norm, y_te_scaled,
                       batch_size, seed, abs_max)


# ─── Shared tensor / DataLoader builder ──────────────────────────────────────

def _to_loaders(X_train, y_train, X_test, y_test, batch_size, seed, abs_max):
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    y_te_t = torch.tensor(y_test,  dtype=torch.float32)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True, generator=g)
    test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size)

    return train_loader, test_loader, abs_max
