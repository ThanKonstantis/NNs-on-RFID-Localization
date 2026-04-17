#!/usr/bin/env bash
# inspect_dataset_sizes.sh
# Reads preprocessed pkl files and prints dataset shapes for every
# trajectory × antenna configuration, before any train/test split.
#
# Usage:
#   ./inspect_dataset_sizes.sh [DATA_DIR]
#
# DATA_DIR defaults to Experiments/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
DATA_DIR="${1:-Experiments}"

"$PYTHON" - "$DATA_DIR" <<'PYEOF'
import sys
import pickle
import numpy as np
from itertools import permutations
from pathlib import Path

data_dir = Path(sys.argv[1])

TRAJECTORIES = {
    "straight": {"pkl": "Experiment_Data_Straight.pkl", "n_ant_3d": [2, 3, 4]},
    "s_path":   {"pkl": "Experiment_Data_S.pkl",        "n_ant_3d": [1, 2, 3, 4]},
    "v_path":   {"pkl": "Experiment_Data_V.pkl",        "n_ant_3d": [1, 2, 3, 4]},
}


def build_2d(data):
    """One sample per (tag, antenna) pair — x_ant, y_ant, phase (3 features)."""
    X, y = [], []
    for tag in data:
        for reading in tag:
            X.append(reading["path"][:, [0, 1, 3]])
            y.append(reading["tag_pos"])
    return np.array(X), np.array(y)


def build_3d(data, n_antennas):
    """Permutations for n<4; fixed order (no permutations) for n=4."""
    X, y = [], []
    for tag in data:
        if n_antennas == 4:
            if len(tag) < 4:
                continue
            X.append(np.hstack([item["path"] for item in tag[:4]]))
            y.append(tag[0]["tag_pos"])
        else:
            for perm in permutations(tag, n_antennas):
                X.append(np.hstack([item["path"] for item in perm]))
                y.append(perm[0]["tag_pos"])
    return np.array(X), np.array(y)


def antenna_distribution(data):
    """Count how many tags have exactly k antenna readings (k=1..4+)."""
    from collections import Counter
    counts = Counter(len(tag) for tag in data)
    return dict(sorted(counts.items()))


any_found = False
for traj, cfg in TRAJECTORIES.items():
    pkl_path = data_dir / cfg["pkl"]
    if not pkl_path.exists():
        print(f"\n[{traj}]  SKIP — {pkl_path} not found")
        continue

    any_found = True
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    dist = antenna_distribution(data)
    total_readings = sum(k * v for k, v in dist.items())

    seq_len = data[0][0]["path"].shape[0]

    print(f"\n{'='*62}")
    print(f"  Trajectory : {traj}")
    print(f"  File       : {pkl_path}")
    print(f"  seq_len    : {seq_len}")
    print(f"  Total tags : {len(data)}  (antenna distribution: {dist})")
    print(f"  Total antenna readings: {total_readings}")
    print(f"{'='*62}")

    # 2D pipeline
    X2, y2 = build_2d(data)
    print(f"  2D pipeline  : X={X2.shape}  y={y2.shape}")

    # 3D pipeline
    for n in cfg["n_ant_3d"]:
        X3, y3 = build_3d(data, n)
        print(f"  3D  n_ant={n}  : X={X3.shape}  y={y3.shape}")

if not any_found:
    print(f"\nNo pkl files found in '{data_dir}'.")
    print("Run build_data.sh first to generate them.")
PYEOF
