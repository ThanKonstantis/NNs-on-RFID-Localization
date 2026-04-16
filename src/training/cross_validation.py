"""K-fold cross-validation loop for RFID localization models."""

import json
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler

from src.evaluation.metrics import eval_model_3d, plot_results
from src.training.callbacks import EarlyStopping
from src.training.data_utils import make_dataloaders, make_cnn_dataloaders, make_rnn_dataloaders
from src.training.trainer import train_test_model


def cross_validate(
    main_data: list,
    holdout_data: list,
    n_antennas: int,
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
    dataloader_fn=None,
) -> dict:
    """Run K-fold cross-validation and final holdout evaluation.

    Permutations are built inside each fold to prevent data leakage.

    If `run_dir` is provided, the trained model, metrics JSON, and distances
    array are saved there automatically.

    Returns a dict with model_name, mean_distance_error_cm, std, distances.
    """
    if dataloader_fn is None:
        dataloader_fn = make_dataloaders

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_losses = []

    # CV folds use a throw-away checkpoint that is cleaned up afterwards
    fold_ckpt = str(run_dir / "_fold_temp.pth") if run_dir else "_fold_temp.pth"

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(main_data)))):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1} / {n_splits} ---")

        train_split = [main_data[i] for i in train_idx]
        val_split = [main_data[i] for i in val_idx]

        X_train, y_train = _build_arrays(train_split, n_antennas)
        X_val, y_val = _build_arrays(val_split, n_antennas)

        train_loader, val_loader, _ = dataloader_fn(
            X_train, y_train, X_val, y_val, n_antennas, batch_size
        )

        model = model_cls(**model_params)
        optimizer = optimizer_cls(model.parameters(), **optimizer_params)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
        early_stopper = EarlyStopping(**{**early_stopper_params, "path": fold_ckpt})

        arr_train, arr_val, best_loss, _ = train_test_model(
            epoch=epochs,
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            device=device,
            test=True,
            verbose=verbose,
        )

        cv_losses.append(best_loss)
        if verbose or run_dir:
            save_path = run_dir / f"fold_{fold_idx + 1}_loss.png" if run_dir else None
            plot_results(np.array(arr_train), np.array(arr_val), save_path=save_path)

    # Clean up fold temp checkpoint
    Path(fold_ckpt).unlink(missing_ok=True)

    if verbose:
        print(f"\nCV done. Mean fold loss: {np.mean(cv_losses):.6f}")

    # --- Final training on all main data, evaluated on holdout ---
    X_main, y_main = _build_arrays(main_data, n_antennas)
    X_holdout, y_holdout = _build_arrays(holdout_data, n_antennas)

    train_loader_full, holdout_loader, scaler = dataloader_fn(
        X_main, y_main, X_holdout, y_holdout, n_antennas, batch_size
    )

    final_ckpt = str(run_dir / "model.pth") if run_dir else early_stopper_params["path"]

    final_model = model_cls(**model_params)
    optimizer = optimizer_cls(final_model.parameters(), **optimizer_params)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    early_stopper = EarlyStopping(**{**early_stopper_params, "path": final_ckpt})

    train_test_model(
        epoch=epochs,
        model=final_model,
        train_loader=train_loader_full,
        test_loader=holdout_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=early_stopper,
        device=device,
        test=True,
        verbose=verbose,
    )

    t_eval_start = time.time()
    result = eval_model_3d(final_model, holdout_loader, scaler, device=device, verbose=verbose,
                           save_path=run_dir / "predictions_3d.png" if run_dir else None)
    evaluation_time_s = round(time.time() - t_eval_start, 4)

    if run_dir is not None:
        _save_run(run_dir, result, run_config, evaluation_time_s)

    return result


def _save_run(run_dir: Path, result: dict, run_config: dict | None,
              evaluation_time_s: float = 0.0) -> None:
    """Save metrics.json and distances.npy to run_dir."""
    distances = result["distances"]

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

    print(f"\nSaved → {run_dir}/")
    print(f"  model.pth  ({(run_dir / 'model.pth').stat().st_size / 1024:.1f} KB)")
    print(f"  metrics.json")
    print(f"  distances.npy")


def _build_arrays(data: list, n_antennas: int):
    dataset, labels = [], []
    for sublist in data:
        for perm in permutations(sublist, n_antennas):
            connected = [item["path"] for item in perm]
            dataset.append(np.hstack(connected))
            labels.append(perm[0]["tag_pos"])
    return np.array(dataset), np.array(labels)
