"""Run the PhaseRelock physics baseline and save results.

Produces the same output structure as train.py so run_experiments.sh can treat
it like any other model:

    saved_models/<timestamp>_phase_relock_<N>ant/
        metrics.json
        distances.npy
        predictions_3d.png        (no model.pth — this is a parameter-free method)

Usage:
    python scripts/run_baseline.py --antennas 3
    python scripts/run_baseline.py --antennas 2 --data Experiments/Experiment_Data.pkl
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.baselines.phase_relock import run_phase_relock
from src.preprocessing.dataset import build_single_arrays, load_experiment_data, split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Run PhaseRelock baseline.")
    parser.add_argument("--antennas", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--data",    default="Experiments/Experiment_Data.pkl")
    parser.add_argument("--holdout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.data} ...")
    experiment_data = load_experiment_data(args.data)
    _, holdout_data = split_data(experiment_data, args.holdout)
    print(f"  {len(experiment_data)} total tags — {len(holdout_data)} holdout")

    X_raw, y_true = build_single_arrays(holdout_data, args.antennas)
    print(f"  {len(X_raw)} holdout samples (one per unique tag position)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path("saved_models") / f"{timestamp}_phase_relock_{args.antennas}ant"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning PhaseRelock ({args.antennas} antennas) ...")
    result = run_phase_relock(
        X_raw, y_true, args.antennas,
        save_path=run_dir / "predictions_3d.png",
    )

    # ── Save metrics.json ─────────────────────────────────────────────────────
    distances   = result["distances"]
    percentiles = np.percentile(distances, [25, 50, 75, 90, 95, 99])
    metrics = {
        "model_class":            "PhaseRelock",
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
        "config": {
            "method":     "phase_relock",
            "n_antennas": args.antennas,
        },
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(run_dir / "distances.npy", distances)

    print(f"\n=== PhaseRelock Results ===")
    print(f"  Mean error : {result['mean_distance_error_cm']:.2f} cm")
    print(f"  Std        : {result['std']:.2f} cm")
    print(f"  Saved to   : {run_dir}/")
    print(f"\nSaved → {run_dir}/")
    print(f"  metrics.json")
    print(f"  distances.npy")
    print(f"  predictions_3d.png")


if __name__ == "__main__":
    main()
