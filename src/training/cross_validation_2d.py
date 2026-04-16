"""K-fold cross-validation for 2-D (single-antenna) RFID localization.

Matches the notebook's cross_validation() behaviour exactly:
  - 5% holdout reserved via train_test_split (random_state=42)
  - KFold cross-validation (shuffle=True, random_state=42) on the remaining 95%
  - Early stopping per fold; epoch count is recorded
  - Final model trained on full 95% for max(epoch_counts) epochs with test=False
    (no early stopping during final training — matches notebook)
  - Holdout set used only for final eval_model_2d call
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.optim import lr_scheduler

from src.evaluation.metrics import plot_results
from src.evaluation.metrics_2d import eval_model_2d
from src.training.callbacks import EarlyStopping
from src.training.data_utils_2d import (
    make_cv_dataloaders_2d,
    make_eval_dataloaders_2d,
    make_cv_dataloaders_2d_rnn,
    make_eval_dataloaders_2d_rnn,
)
from src.training.trainer import train_test_model


def cross_validate_2d(
    input_array: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    model_cls,
    model_params: dict,
    loss_fn,
    optimizer_cls,
    optimizer_params: dict,
    scheduler_params: dict,
    early_stopper_params: dict,
    epochs: int,
    device: str = "cpu",
    batch_size: int = 32,
    verbose: bool = True,
    run_dir: Path | None = None,
    run_config: dict | None = None,
    print_every: int = 10,
) -> dict:
    """Run K-fold CV and final holdout evaluation on 2-D numpy data.

    Parameters
    ----------
    input_array:  (N, seq_len, n_features) raw sensor data
    labels:       (N, >=2) tag positions in cm; only the first 2 columns are used
    ...           (same interface as cross_validate in cross_validation.py)

    Returns a dict with model_name, mean_distance_error_cm, std, distances.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    # ── 5% holdout split (matches notebook) ──────────────────────────────────
    X_main, X_holdout, y_main_full, y_holdout_full = train_test_split(
        input_array, labels, test_size=0.10, random_state=42
    )
    y_main    = y_main_full[:, :2].copy()
    y_holdout = y_holdout_full[:, :2].copy()

    # ── Select dataloader variant based on model input type ──────────────────
    input_type = getattr(model_cls, "input_type", "flat")
    if input_type == "rnn":
        cv_loader_fn   = make_cv_dataloaders_2d_rnn
        eval_loader_fn = make_eval_dataloaders_2d_rnn
    else:
        cv_loader_fn   = make_cv_dataloaders_2d
        eval_loader_fn = make_eval_dataloaders_2d

    # ── K-fold cross-validation ───────────────────────────────────────────────
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_losses   = []
    epoch_array = []

    fold_ckpt = str(run_dir / "_fold_temp.pth") if run_dir else "_fold_temp_2d.pth"

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_main)):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1} / {n_splits} ---")

        train_loader, val_loader, _ = cv_loader_fn(
            X_main, y_main, train_idx, val_idx, batch_size=batch_size
        )

        model     = model_cls(**model_params)
        optimizer = optimizer_cls(model.parameters(), **optimizer_params)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        stopper   = EarlyStopping(**{**early_stopper_params, "path": fold_ckpt})

        arr_train, arr_val, best_loss, epoch_stop = train_test_model(
            epoch=epochs,
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=stopper,
            device=device,
            test=True,
            verbose=verbose,
            print_every=print_every,
        )

        cv_losses.append(best_loss)
        epoch_array.append(epoch_stop)

        save_path = run_dir / f"fold_{fold_idx + 1}_loss.png" if run_dir else None
        plot_results(np.array(arr_train), np.array(arr_val), save_path=save_path)

    Path(fold_ckpt).unlink(missing_ok=True)

    mean_cv = np.mean(cv_losses)
    final_epochs = int(np.max(epoch_array))
    if verbose:
        print(f"\nCV done. Mean fold loss: {mean_cv:.6f}")
        print(f"Final model will train for {final_epochs} epochs (max of fold stops).")

    # ── Final model: trained on all main data, no early stopping ─────────────
    # Matches notebook: train_test_model(..., test=False)
    train_loader_full, holdout_loader, scaler = eval_loader_fn(
        X_main.copy(), X_holdout.copy(), y_main.copy(), y_holdout.copy(),
        batch_size=batch_size,
    )

    final_model = model_cls(**model_params)
    optimizer   = optimizer_cls(final_model.parameters(), **optimizer_params)
    scheduler   = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    train_test_model(
        epoch=final_epochs,
        model=final_model,
        train_loader=train_loader_full,
        test_loader=holdout_loader,   # passed but ignored (test=False)
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=None,
        device=device,
        test=False,
        verbose=verbose,
        print_every=print_every,
    )

    if run_dir:
        torch.save(final_model.state_dict(), run_dir / "model.pth")

    # ── Final evaluation on holdout ───────────────────────────────────────────
    t_eval = time.time()
    result = eval_model_2d(
        final_model, holdout_loader, scaler,
        device=device, verbose=verbose,
        save_path=run_dir / "predictions_2d.pdf" if run_dir else None,
    )
    evaluation_time_s = round(time.time() - t_eval, 4)

    if run_dir is not None:
        _save_run_2d(run_dir, result, run_config, evaluation_time_s)

    return result


def _save_run_2d(run_dir: Path, result: dict, run_config: dict | None,
                 evaluation_time_s: float = 0.0) -> None:
    distances   = result["distances"]
    percentiles = np.percentile(distances, [25, 50, 75, 90, 95, 99])
    metrics = {
        "model_class":            result["model_name"],
        "mean_distance_error_cm": round(result["mean_distance_error_cm"], 4),
        "std_cm":                 round(result["std"], 4),
        "percentiles_cm": {
            "p25": round(float(percentiles[0]), 4),
            "p50": round(float(percentiles[1]), 4),
            "p75": round(float(percentiles[2]), 4),
            "p90": round(float(percentiles[3]), 4),
            "p95": round(float(percentiles[4]), 4),
            "p99": round(float(percentiles[5]), 4),
        },
        "evaluation_time_s": evaluation_time_s,
    }
    if run_config:
        metrics["config"] = run_config

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(run_dir / "distances.npy", distances)

    model_path = run_dir / "model.pth"
    size_kb = model_path.stat().st_size / 1024 if model_path.exists() else 0
    print(f"\nSaved → {run_dir}/")
    print(f"  model.pth  ({size_kb:.1f} KB)")
    print(f"  metrics.json")
    print(f"  distances.npy")
