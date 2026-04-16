"""Build train/test datasets from the processed experiment pickle file."""

import pickle
import random
from itertools import permutations
from pathlib import Path

import numpy as np


def load_experiment_data(pkl_path: str | Path, seed: int = 42) -> list:
    """Load and shuffle the experiment data pickle."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    random.seed(seed)
    random.shuffle(data)
    return data


def split_data(data: list, holdout_ratio: float = 0.1):
    """Split data into main (train+val) and holdout (test) sets."""
    n_holdout = int(holdout_ratio * len(data))
    main_data = data[:-n_holdout] if n_holdout > 0 else data
    holdout_data = data[-n_holdout:] if n_holdout > 0 else []
    return main_data, holdout_data


def build_single_arrays(data: list, n_antennas: int) -> tuple[np.ndarray, np.ndarray]:
    """One sample per unique tag: take the first n_antennas antennas, no permutations.

    Use this for parameter-free baselines where antenna order does not matter.
    Entries with fewer than n_antennas readings are silently skipped (same
    behaviour as build_permutation_arrays which uses itertools.permutations).
    Returns (input_array, labels) with shape (N_tags, interp_length, 4*n_antennas)
    and (N_tags, 3).
    """
    dataset, label_list = [], []
    for sublist in data:
        if len(sublist) < n_antennas:
            continue
        paths = np.hstack([sublist[k]["path"] for k in range(n_antennas)])
        dataset.append(paths)
        label_list.append(sublist[0]["tag_pos"])
    return np.array(dataset), np.array(label_list)


def build_2d_arrays(data: list) -> tuple[np.ndarray, np.ndarray]:
    """Build 2-D single-antenna training arrays from the pkl data.

    Each tag contributes one sample per antenna reading (no permutations).
    The antenna path has shape (seq_len, 4) = [x_ant, y_ant, z_ant, phase].
    Labels are (N, 3) = [x_tag, y_tag, z_tag] — callers slice [:, :2] for 2-D.

    Returns
    -------
    input_array : (N, seq_len, 4)
    labels      : (N, 3)
    """
    dataset, labels = [], []
    for tag_readings in data:
        for reading in tag_readings:
            dataset.append(reading["path"][:, [0, 1, 3]])  # x_ant, y_ant, phase (drop z_ant)
            labels.append(reading["tag_pos"])
    return np.array(dataset), np.array(labels)


def build_permutation_arrays(data: list, n_antennas: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate all permutations of `n_antennas` antenna paths for each tag and stack them.

    Returns (input_array, labels) where:
      input_array : (N, interp_length, 4 * n_antennas)
      labels      : (N, 3)  — [x, y, z] tag position
    """
    dataset = []
    label_list = []

    for sublist in data:
        for perm in permutations(sublist, n_antennas):
            connected_paths = [item["path"] for item in perm]
            paths = np.hstack(connected_paths)
            dataset.append(paths)
            label_list.append(perm[0]["tag_pos"])

    return np.array(dataset), np.array(label_list)
