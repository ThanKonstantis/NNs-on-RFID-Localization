"""Train an RFID localization model with K-fold cross-validation.

Each run is saved automatically to saved_models/<timestamp>_<model>_<N>ant/
containing model.pth, metrics.json, and distances.npy.

Usage examples:
    python scripts/train.py --antennas 3
    python scripts/train.py --antennas 2 --model relu --epochs 200
    python scripts/train.py --antennas 4 --model leaky_relu --epochs 300
    python scripts/train.py --list-models
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.networks import MODEL_REGISTRY
from src.preprocessing.dataset import load_experiment_data, split_data
from src.training.cross_validation import cross_validate


OPTIMIZER_MAP = {
    "adam":    torch.optim.Adam,
    "adamw":   torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "sgd":     torch.optim.SGD,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train RFID localization model.")
    parser.add_argument("--antennas", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--model", default="leaky_relu",
                        help="Architecture name. Use --list-models to see all.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=list(OPTIMIZER_MAP), default="adam")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=90)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--data", default="Experiments/Experiment_Data.pkl")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


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

    # Load data
    print(f"Loading data from {args.data} ...")
    experiment_data = load_experiment_data(args.data)
    main_data, holdout_data = split_data(experiment_data, args.holdout)
    print(f"  {len(experiment_data)} tags — {len(main_data)} main / {len(holdout_data)} holdout")

    # Compute input/output sizes
    trial = experiment_data[0][0]["path"]
    input_len = trial.shape[0] * trial.shape[1] * args.antennas
    output_len = len(experiment_data[0][0]["tag_pos"])
    print(f"  input_len={input_len}, output_len={output_len}")

    # Model params
    model_cls = MODEL_REGISTRY[args.model]
    model_params = {"input_size": input_len, "output_size": output_len}

    # Auto-generate run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("saved_models") / f"{timestamp}_{args.model}_{args.antennas}ant"

    # Config dict saved into metrics.json
    run_config = {
        "model":        args.model,
        "n_antennas":   args.antennas,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "lr":           args.lr,
        "optimizer":    args.optimizer,
        "folds":        args.folds,
        "patience":     args.patience,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
    }

    # Training config
    optimizer_cls = OPTIMIZER_MAP[args.optimizer]
    optimizer_params = {"lr": args.lr}
    if args.optimizer == "sgd":
        optimizer_params["momentum"] = 0.9

    scheduler_params = {"mode": "min", "factor": 0.1, "patience": 10, "min_lr": 1e-5}
    early_stopper_params = {
        "patience":  args.patience,
        "min_delta": 1e-5,
        "verbose":   not args.quiet,
        "path":      "_temp.pth",   # overridden inside cross_validate per fold/final
    }

    print(f"\nTraining {model_cls.__name__} | antennas={args.antennas} | "
          f"epochs={args.epochs} | folds={args.folds}")
    print(f"Run will be saved to: {run_dir}/\n")

    result = cross_validate(
        main_data=main_data,
        holdout_data=holdout_data,
        n_antennas=args.antennas,
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
    )

    print("\n=== Final Results ===")
    print(f"  Model:      {result['model_name']}")
    print(f"  Mean error: {result['mean_distance_error_cm']:.2f} cm")
    print(f"  Std:        {result['std']:.2f} cm")
    print(f"  Saved to:   {run_dir}/")


if __name__ == "__main__":
    main()
