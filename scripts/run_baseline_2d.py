"""Run the 2-D Phase Relock baseline on the single-antenna dataset.

Loads the raw numpy data, reserves a 5% holdout (same split as the NN experiments),
runs the nonlinear optimisation, and saves metrics.json + distances.npy.

Usage:
    python scripts/run_baseline_2d.py
    python scripts/run_baseline_2d.py --data Experiments/Raw_Data_Single_Antenna_0
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from src.baselines.phase_relock_2d import run_phase_relock_2d

DEFAULT_DATA = "Experiments/Raw_Data_Single_Antenna_0"


def parse_args():
    parser = argparse.ArgumentParser(description="2-D Phase Relock baseline.")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Directory with final_tensor.npy and final_labels.npy.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)

    if not data_dir.is_dir():
        print(f"ERROR: Data directory '{data_dir}' not found.")
        sys.exit(1)

    print(f"Loading 2D data from {data_dir} ...")
    info_tensor = np.load(data_dir / "final_tensor.npy")
    rfid_label  = np.load(data_dir / "final_labels.npy")
    print(f"  info_tensor : {info_tensor.shape}")
    print(f"  rfid_label  : {rfid_label.shape}")

    # Use the same 5% holdout split as the NN cross_validate_2d
    _, X_holdout, _, y_holdout = train_test_split(
        info_tensor, rfid_label, test_size=0.10, random_state=42
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path("saved_models") / f"{timestamp}_phase_relock_2D"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning Phase Relock 2D on {len(X_holdout)} holdout samples ...")
    t0 = time.time()
    result = run_phase_relock_2d(
        X_raw=X_holdout,
        y_true_cm=y_holdout,
        save_path=run_dir / "predictions_2d.png",
    )
    elapsed = round(time.time() - t0, 4)

    print(f"\nEvaluation time: {elapsed} s  |  {len(X_holdout)} samples")
    print(f"Mean error: {result['mean_distance_error_cm']:.2f} cm")
    print(f"Std:        {result['std']:.2f} cm")

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
        "evaluation_time_s": elapsed,
        "config": {
            "data": str(data_dir),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(run_dir / "distances.npy", distances)

    print(f"\nSaved → {run_dir}/")
    print(f"  metrics.json")
    print(f"  distances.npy")


if __name__ == "__main__":
    main()
