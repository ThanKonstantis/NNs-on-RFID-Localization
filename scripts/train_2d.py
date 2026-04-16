"""Train a 2-D RFID localization model with K-fold cross-validation.

Loads data from the same Experiment_Data.pkl used by the 3-D pipeline.
Each tag contributes one sample per antenna reading (no permutations).
Antenna path columns: [x_ant, y_ant, z_ant, phase].
Labels used: [x_tag, y_tag] where y_tag is the radial distance r from the
antenna path to the tag.

Each run is saved to saved_models/<timestamp>_<model>_2D/ containing
model.pth, metrics.json, distances.npy, and fold loss plots.

Usage examples:
    python scripts/train_2d.py
    python scripts/train_2d.py --model relu --epochs 200
    python scripts/train_2d.py --model mlp --hidden 1540,1024,512,256,128 --batch-size 64
    python scripts/train_2d.py --model simple --optimizer rmsprop --epochs 100
    python scripts/train_2d.py --model leaky_relu_drop --lr 1e-2 --weight-decay 1e-4 --epochs 350
    python scripts/train_2d.py --list-models
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.models.networks import MODEL_REGISTRY
from src.preprocessing.dataset import load_experiment_data, build_2d_arrays
from src.training.cross_validation_2d import cross_validate_2d


OPTIMIZER_MAP = {
    "adam":    torch.optim.Adam,
    "adamw":   torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "sgd":     torch.optim.SGD,
}

DEFAULT_DATA = "Experiments/Experiment_Data.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 2-D RFID localization model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="leaky_relu",
                        help="Architecture name. Use --list-models to see all.")
    parser.add_argument("--hidden",
                        help="Hidden unit sizes, comma-separated (e.g. 128 or 1540,1024,512). "
                             "Overrides model default.")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight-decay",type=float, default=0.0,
                        help="L2 regularisation for Adam/AdamW.")
    parser.add_argument("--optimizer",   choices=list(OPTIMIZER_MAP), default="adam")
    parser.add_argument("--folds",       type=int,   default=5)
    parser.add_argument("--patience",    type=int,   default=90)
    parser.add_argument("--data",        default=DEFAULT_DATA,
                        help="Path to Experiment_Data.pkl.")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--quiet",       action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def _parse_hidden(hidden_str):
    """Parse '128' → (128,) or '1540,1024,512' → (1540, 1024, 512)."""
    return tuple(int(x) for x in hidden_str.split(","))


def main():
    args = parse_args()

    if args.list_models:
        print("Available models:")
        for name in sorted(MODEL_REGISTRY):
            print(f"  {name:20s} → {MODEL_REGISTRY[name].__name__}")
        return

    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model '{args.model}'. Use --list-models to see options.")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file '{data_path}' not found.")
        sys.exit(1)

    print(f"Loading data from {data_path} ...")
    experiment_data = load_experiment_data(data_path)
    info_tensor, rfid_label = build_2d_arrays(experiment_data)
    print(f"  Samples     : {len(info_tensor)}  ({len(experiment_data)} tags)")
    print(f"  info_tensor : {info_tensor.shape}  [x_ant, y_ant, z_ant, phase]")
    print(f"  rfid_label  : {rfid_label.shape}   [x_tag, y_tag(r), z_tag]")

    seq_len    = info_tensor.shape[1]
    n_features = info_tensor.shape[2]
    output_len = 2   # 2-D: X, Y only

    # ── Build model params ────────────────────────────────────────────────────
    model_cls  = MODEL_REGISTRY[args.model]
    input_type = getattr(model_cls, "input_type", "flat")

    # Resolve hidden_units override
    hidden_override = _parse_hidden(args.hidden) if args.hidden else None

    if input_type == "rnn":
        model_params = {"input_size": n_features, "output_size": output_len}
        if hidden_override:
            model_params["hidden_size"] = hidden_override[0]
    elif input_type == "cnn":
        model_params = {"input_channels": n_features, "output_size": output_len}
    else:
        flat_input = seq_len * n_features
        model_params = {"input_size": flat_input, "output_size": output_len}
        if hidden_override:
            import inspect
            sig = inspect.signature(model_cls.__init__)
            if "hidden_units" in sig.parameters:
                default_val = sig.parameters["hidden_units"].default
                # FlexibleMLP default is a tuple → keep as tuple
                # All other models use a single int
                if isinstance(default_val, (list, tuple)):
                    model_params["hidden_units"] = hidden_override
                else:
                    model_params["hidden_units"] = hidden_override[0]

    if not args.quiet:
        print(f"  Model params: {model_params}")

    # ── Output directory ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label     = args.model
    if args.hidden:
        label += f"_h{args.hidden.replace(',', '-')}"
    run_dir = Path("saved_models") / f"{timestamp}_{label}_2D"

    run_config = {
        "model":        args.model,
        "hidden":       args.hidden or "default",
        "mode":         "2D",
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "optimizer":    args.optimizer,
        "folds":        args.folds,
        "patience":     args.patience,
        "data":         str(data_path),
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
    }

    # ── Training config ───────────────────────────────────────────────────────
    optimizer_cls    = OPTIMIZER_MAP[args.optimizer]
    optimizer_params = {"lr": args.lr}
    if args.weight_decay > 0.0 and args.optimizer in ("adam", "adamw"):
        optimizer_params["weight_decay"] = args.weight_decay
    if args.optimizer == "sgd":
        optimizer_params["momentum"] = 0.9

    scheduler_params     = {"mode": "min", "factor": 0.1, "patience": 10, "min_lr": 1e-5}
    early_stopper_params = {
        "patience":  args.patience,
        "min_delta": 1e-5,
        "verbose":   not args.quiet,
        "path":      "_temp_2d.pth",
    }

    print(f"\nTraining {model_cls.__name__} | 2D | "
          f"epochs={args.epochs} | folds={args.folds} | "
          f"lr={args.lr} | optimizer={args.optimizer}")
    print(f"Run will be saved to: {run_dir}/\n")

    result = cross_validate_2d(
        input_array=info_tensor,
        labels=rfid_label,
        n_splits=args.folds,
        model_cls=model_cls,
        model_params=model_params,
        loss_fn=nn.MSELoss(),
        optimizer_cls=optimizer_cls,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        early_stopper_params=early_stopper_params,
        epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        run_dir=run_dir,
        run_config=run_config,
        print_every=1 if input_type == "rnn" else 10,
    )

    print("\n=== Final Results ===")
    print(f"  Model:      {result['model_name']}")
    print(f"  Mean error: {result['mean_distance_error_cm']:.2f} cm")
    print(f"  Std:        {result['std']:.2f} cm")
    print(f"  Saved to:   {run_dir}/")


if __name__ == "__main__":
    main()
