"""Process raw RFID measurements into the Experiment_Data.pkl used for NN training.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --experiment "Straight " --interp-length 385 --size-threshold 10240
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.preprocessing.interpolation import lin_interpolation
from src.preprocessing.transforms import rotate_points_to_zero, rotate_with_respect


def transform_coordinates(root_folder: Path) -> None:
    """Step 1 — rotate all trajectories to a canonical frame and save processed xlsx files."""
    print("Step 1: Transforming coordinates...")
    for folder_path, _, _ in os.walk(root_folder):
        if "Antenna_" in folder_path:
            continue
        file_tag = os.path.join(folder_path, "Transformed_Coordinates.xlsx")
        try:
            rfid_df = pd.read_excel(file_tag)
        except FileNotFoundError:
            continue

        folders = ["Antenna_1", "Antenna_2", "Antenna_3", "Antenna_4"]

        for tag in rfid_df["EPC_TAG"].unique():
            row = rfid_df[rfid_df["EPC_TAG"] == tag]
            rfid_x = row["X"].values[0]
            rfid_y = row["Y"].values[0]

            largest_file, largest_size = None, -1
            for folder in folders:
                unwrapped = os.path.join(folder_path, folder, "unwrapped_measurements")
                if os.path.exists(unwrapped):
                    for f in os.listdir(unwrapped):
                        if f == f"{tag}.txt":
                            fp = os.path.join(unwrapped, f)
                            sz = os.path.getsize(fp)
                            if sz > largest_size:
                                largest_size, largest_file = sz, fp

            if not largest_file:
                continue

            df = pd.read_csv(largest_file, header=None,
                             names=["time", "X", "Y", "Z", "Power", "Phase", "Phase_Unwrapped"])
            X, Y = df["X"].values, df["Y"].values
            X_rot, Y_rot, rfid_pos, rot_angle = rotate_points_to_zero(X, Y, [rfid_x * 100, rfid_y * 100])

            df["X_new"], df["Y_new"] = X_rot, Y_rot
            rfid_df.loc[rfid_df["EPC_TAG"] == tag, "X_new"] = rfid_pos[0]
            rfid_df.loc[rfid_df["EPC_TAG"] == tag, "Y_new"] = rfid_pos[1]
            rfid_df.to_excel(file_tag, index=False)
            df.to_excel(largest_file.replace(".txt", "_processed.xlsx"), index=False)

            # Rotate secondary antennas
            for folder in folders:
                unwrapped = os.path.join(folder_path, folder, "unwrapped_measurements")
                if not os.path.exists(unwrapped):
                    continue
                if f"{tag}_processed.xlsx" in os.listdir(unwrapped):
                    continue
                for f in os.listdir(unwrapped):
                    if f == f"{tag}.txt":
                        fp = os.path.join(unwrapped, f)
                        df1 = pd.read_csv(fp, header=None,
                                          names=["time", "X", "Y", "Z", "Power", "Phase", "Phase_Unwrapped"])
                        X1, Y1 = df1["X"].values, df1["Y"].values
                        X_rot1, Y_rot1 = rotate_with_respect(X[0], Y[0], rot_angle, X1, Y1)
                        df1["X_new"], df1["Y_new"] = X_rot1, Y_rot1
                        df1.to_excel(fp.replace(".txt", "_processed_smaller.xlsx"), index=False)

    print("  Done.")


def build_tensor(root_folder: Path, experiment: str, interp_length: int,
                 size_threshold: int, output_path: Path) -> None:
    """Step 2 — interpolate and save Experiment_Data.pkl."""
    print("Step 2: Building interpolated tensor...")
    sel_cols = ["X_new", "Y_new", "Z", "Phase_Unwrapped"]
    final_tensor = []

    for folder_path, _, _ in os.walk(root_folder):
        if experiment not in folder_path:
            continue
        if "Antenna_" in folder_path:
            continue

        file_tag = os.path.join(folder_path, "Transformed_Coordinates.xlsx")
        try:
            rfid_df = pd.read_excel(file_tag)
        except FileNotFoundError:
            continue

        folders = ["Antenna_1", "Antenna_2", "Antenna_3", "Antenna_4"]

        for tag in rfid_df["EPC_TAG"].unique():
            row = rfid_df[rfid_df["EPC_TAG"] == tag]
            rfid_x = row["X_new"].values[0].item()
            rfid_y = row["Y_new"].values[0].item()
            rfid_z = row["Z"].values[0].item() * 100

            antenna_data = []
            ant_num = 0
            for folder in folders:
                unwrapped = os.path.join(folder_path, folder, "unwrapped_measurements")
                if not os.path.exists(unwrapped):
                    continue
                ant_num += 1
                for f in os.listdir(unwrapped):
                    if f in (f"{tag}_processed.xlsx", f"{tag}_processed_smaller.xlsx"):
                        fp = os.path.join(unwrapped, f)
                        if os.path.getsize(fp) < size_threshold:
                            continue
                        tag_df = pd.read_excel(fp)
                        info = tag_df[sel_cols].to_numpy()
                        res = lin_interpolation(info, interp_length)
                        antenna_data.append({
                            "Antenna": ant_num,
                            "path": res,
                            "tag_name": tag,
                            "tag_pos": [rfid_x, rfid_y, rfid_z],
                        })
            final_tensor.append(antenna_data)

    with open(output_path, "wb") as f:
        pickle.dump(final_tensor, f)
    print(f"  Saved {len(final_tensor)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw RFID measurements.")
    parser.add_argument("--measurements-dir", default="Experiments/Measurements",
                        help="Path to the Measurements folder (default: Experiments/Measurements)")
    parser.add_argument("--experiment", default="Straight ",
                        help="Substring to filter experiment folders (default: 'Straight ')")
    parser.add_argument("--interp-length", type=int, default=385,
                        help="Number of interpolation points (default: 385)")
    parser.add_argument("--size-threshold", type=int, default=10240,
                        help="Minimum file size in bytes (default: 10240 = 10 KB)")
    parser.add_argument("--output", default="Experiments/Experiment_Data.pkl",
                        help="Output pickle path (default: Experiments/Experiment_Data.pkl)")
    parser.add_argument("--skip-transform", action="store_true",
                        help="Skip coordinate transformation step (use if already done)")
    args = parser.parse_args()

    root = Path(args.measurements_dir)
    output = Path(args.output)

    if not args.skip_transform:
        transform_coordinates(root)

    build_tensor(root, args.experiment, args.interp_length, args.size_threshold, output)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
