"""Simulation data-generation routines."""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.simulation.physics import Antenna, Robot, Tag, compute_phase, load_antenna_pattern

FREQ = 866e6
C = 299_792_458.0
LAMBDA = C / FREQ
PT_DBM = 30.0
G_R_DBM = 1.0
STEP_MU = 0.0038
STEP_SIGMA = 0.001834


def simulate_straight_path_3d(
    pattern_path: str,
    antenna_heights: list[float],
    path_length: float = 4.0,
    x_tag: float = 2.0,
    y_steps: np.ndarray | None = None,
    z_steps: np.ndarray | None = None,
    output_path: str | None = None,
) -> list:
    """Simulate robot moving along the x-axis with multiple antennas at different heights.

    Returns a list of sub-lists (one per tag location), each sub-list containing
    one dict per antenna with keys: Antenna, path, tag_name, tag_pos.
    Compatible with the experiment data format used for NN training.
    """
    if y_steps is None:
        y_steps = np.linspace(0.1, 3.0, 30)
    if z_steps is None:
        z_steps = np.linspace(0.1, 2.0, 30)

    gain_theta, gain_phi = load_antenna_pattern(pattern_path)
    antennas = [Antenna(z, 4.0, gain_theta, gain_phi) for z in antenna_heights]

    data_list = []

    for y_tag in y_steps:
        for z_tag in z_steps:
            tag_entry = []
            for ant_idx, antenna in enumerate(antennas):
                p_start = np.array([0.0, 0.0, antenna.z])
                p_end = np.array([path_length, 0.0, antenna.z])
                tag = Tag(np.array([x_tag, y_tag, z_tag]))

                phases, robot_pos = [], []
                x = 0.0
                while x < path_length:
                    x += np.random.normal(STEP_MU, STEP_SIGMA)
                    pos = np.array([p_start[0] + x, 0.0, antenna.z])
                    robot_pos.append(pos.tolist())
                    phase = compute_phase(tag, antenna, pos, LAMBDA)
                    phases.append(phase)

                phases_arr = np.array(phases)
                robot_arr = np.array(robot_pos)
                # Combine: [x, y, z, phase_unwrapped] — subtract first phase for offset removal
                combined = np.column_stack([robot_arr, phases_arr - phases_arr[0]])
                tag_entry.append({
                    "Antenna": ant_idx + 1,
                    "path": combined,
                    "tag_name": f"y{y_tag:.3f}_z{z_tag:.3f}",
                    "tag_pos": [x_tag, y_tag, z_tag],
                })
            data_list.append(tag_entry)

    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(data_list, f)
        print(f"Saved {len(data_list)} samples to {output_path}")

    return data_list


def interpolate_data(data_list: list, length: int = 385) -> list:
    """Interpolate each antenna path to a fixed number of samples."""
    for tag_entry in data_list:
        for ant_data in tag_entry:
            path = np.array(ant_data["path"])
            x = path[:, 0]
            xd = np.linspace(x[0], x[-1], length)
            interped = [xd]
            for col in range(1, path.shape[1]):
                interped.append(
                    interp1d(x, path[:, col], kind="linear",
                             bounds_error=False, fill_value="extrapolate")(xd)
                )
            ant_data["path"] = np.vstack(interped).T
    return data_list
