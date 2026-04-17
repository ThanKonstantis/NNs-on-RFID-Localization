"""Build final_tensor.npy and final_labels.npy for 2-D (single-antenna) experiments.

Replicates the data-generation cells from notebooks/Processed_Measurements.ipynb.

For each measurement folder that matches the experiment filter:
  - Reads Transformed_Coordinates.xlsx for tag ground-truth positions
  - Finds the matching *_processed2D.xlsx file in unwrapped_measurements/
  - Skips files below the size threshold (likely empty / corrupted)
  - Interpolates each trajectory to a fixed length along arc-length
  - Saves final_tensor.npy  (N, interp_length, 3)  [x_rob, y_rob, phase]
          final_labels.npy  (N, 3)                  [x_tag, y_tag, z_tag]
  into a new auto-indexed Raw_Data_Single_Antenna_<N> folder.

Usage:
    python scripts/build_data_2d.py
    python scripts/build_data_2d.py --measurements-dir Experiments/Measurements
    python scripts/build_data_2d.py --interp-length 385 --size-threshold 10240
    python scripts/build_data_2d.py --output-dir Experiments/Raw_Data_Single_Antenna_0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.preprocessing.interpolation import lin_interpolation


EXPERIMENT_FILTER = "Straight "
DEFAULT_MEASUREMENTS = "Experiments/Measurements"
DEFAULT_SIZE_THRESHOLD = 10240   # 10 KB

# p75 of per-sample trajectory lengths, computed from raw measurements
_INTERP_LENGTH_P75 = {
    "straight": 368,
    "s":        468,
    "v":        402,
}

def _default_interp_length(experiment: str) -> int:
    exp_lower = experiment.lower()
    if "straight" in exp_lower:
        return _INTERP_LENGTH_P75["straight"]
    if "s" in exp_lower:
        return _INTERP_LENGTH_P75["s"]
    if "v" in exp_lower:
        return _INTERP_LENGTH_P75["v"]
    return 368  # fallback


def parse_args():
    parser = argparse.ArgumentParser(description="Build 2D single-antenna dataset.")
    parser.add_argument("--measurements-dir", default=DEFAULT_MEASUREMENTS,
                        help="Root folder containing the measurement sub-folders.")
    parser.add_argument("--experiment", default=EXPERIMENT_FILTER,
                        help="Substring filter for experiment folders (default: 'Straight ').")
    parser.add_argument("--interp-length", type=int, default=None,
                        help="Number of interpolation points per trajectory. Defaults to "
                             "the p75 for the selected path (straight=368, s=468, v=402).")
    parser.add_argument("--size-threshold", type=int, default=DEFAULT_SIZE_THRESHOLD,
                        help="Minimum file size in bytes to include a reading (default: 10240).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory path. If omitted, auto-increments "
                             "Raw_Data_Single_Antenna_<N> inside --measurements-dir's parent.")
    return parser.parse_args()


def _next_output_dir(base_dir: Path) -> Path:
    """Auto-increment Raw_Data_Single_Antenna_<N> inside base_dir."""
    existing = []
    prefix = "Raw_Data_Single_Antenna_"
    for folder in base_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            try:
                existing.append(int(folder.name[len(prefix):]))
            except ValueError:
                pass
    new_index = (max(existing) + 1) if existing else 0
    return base_dir / f"{prefix}{new_index}"


def build(measurements_dir: Path, experiment: str, interp_length: int,
          size_threshold: int, output_dir: Path | None) -> Path:

    final_tensor = []
    final_labels = []
    skipped = 0

    sel_cols = ["X_new", "Y_new", "Phase_Unwrapped"]

    for folder_path, _, _ in os.walk(measurements_dir):
        folder_path = Path(folder_path)

        # Only process folders that belong to the target experiment
        if experiment not in str(folder_path):
            continue

        # Only process antenna sub-folders (not the top-level experiment folder)
        if "Antenna_" not in folder_path.name:
            continue

        coord_file = folder_path / "Transformed_Coordinates.xlsx"
        if not coord_file.exists():
            continue

        try:
            rfid_df = pd.read_excel(coord_file)
        except Exception as e:
            print(f"  WARNING: could not read {coord_file}: {e}")
            continue

        unwrapped_dir = folder_path / "unwrapped_measurements"
        if not unwrapped_dir.exists():
            continue

        for tag in rfid_df["EPC_TAG"].unique():
            row     = rfid_df[rfid_df["EPC_TAG"] == tag]
            rfid_x  = row["X_new"].values[0]
            rfid_y  = row["Y_new"].values[0]
            rfid_z  = row["Z"].values[0] * 100   # m → cm (matches notebook)

            meas_file = unwrapped_dir / f"{tag}_processed2D.xlsx"
            if not meas_file.exists():
                continue

            if meas_file.stat().st_size < size_threshold:
                skipped += 1
                continue

            try:
                tag_df = pd.read_excel(meas_file)
            except Exception as e:
                print(f"  WARNING: could not read {meas_file}: {e}")
                continue

            info = tag_df[sel_cols].to_numpy()
            res  = lin_interpolation(info, interp_length)

            final_tensor.append(res)
            final_labels.append([rfid_x, rfid_y, rfid_z])

    if not final_tensor:
        print("ERROR: No samples found. Check --measurements-dir and --experiment filter.")
        sys.exit(1)

    final_tensor = np.array(final_tensor)
    final_labels = np.array(final_labels)

    print(f"Samples   : {len(final_tensor)}  (skipped {skipped} below size threshold)")
    print(f"Tensor    : {final_tensor.shape}   [x_rob, y_rob, phase]")
    print(f"Labels    : {final_labels.shape}   [x_tag, y_tag, z_tag]")

    # Resolve output directory
    if output_dir is None:
        output_dir = _next_output_dir(measurements_dir.parent)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "final_tensor.npy", final_tensor)
    np.save(output_dir / "final_labels.npy", final_labels)

    print(f"\nSaved → {output_dir}/")
    print(f"  final_tensor.npy")
    print(f"  final_labels.npy")
    return output_dir


def main():
    args = parse_args()
    measurements_dir = Path(args.measurements_dir)

    if not measurements_dir.is_dir():
        print(f"ERROR: '{measurements_dir}' not found.")
        sys.exit(1)

    interp_length = args.interp_length if args.interp_length is not None \
        else _default_interp_length(args.experiment)

    print(f"Building 2D dataset")
    print(f"  Measurements : {measurements_dir}")
    print(f"  Experiment   : '{args.experiment}'")
    print(f"  Interp length: {interp_length}")
    print(f"  Size threshold: {args.size_threshold} bytes")
    print()

    out = build(
        measurements_dir=measurements_dir,
        experiment=args.experiment,
        interp_length=interp_length,
        size_threshold=args.size_threshold,
        output_dir=args.output_dir,
    )

    print(f"\nDone. Pass this path to the experiment script:")
    print(f"  ./run_experiments_2D.sh my_run --data {out}")


if __name__ == "__main__":
    main()
