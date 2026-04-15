"""Physical simulation classes for RFID robot localization."""

import random

import numpy as np
import pandas as pd


class Antenna:
    def __init__(self, z: float, G_dBm: float, theta_vector: np.ndarray, phi_vector: np.ndarray):
        self.z = z
        self.G_dBm = G_dBm
        self.theta_vector = theta_vector
        self.phi_vector = phi_vector
        self.phase_bias = random.uniform(0, 2 * np.pi)

    def get_gain(self, theta_rad: float, phi_rad: float) -> float:
        theta_deg = round(theta_rad * 180 / np.pi) % 360
        phi_deg = round(phi_rad * 180 / np.pi) % 360
        return self.theta_vector[theta_deg] + self.phi_vector[phi_deg] + self.G_dBm


class Tag:
    def __init__(self, pos: np.ndarray):
        self.position = pos


class Robot:
    def __init__(self, pos: np.ndarray, antennas: list):
        self.position = pos
        self.antennas = antennas
        self.num_of_antennas = len(antennas)

    def update_pos(self, pos):
        self.position = np.asarray(pos)


def load_antenna_pattern(pattern_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load gain pattern from an Excel file."""
    pattern = pd.read_excel(pattern_path)
    return pattern["Gain_t(dB)"].to_numpy(), pattern["Gain_p(dB)"].to_numpy()


def compute_phase(tag: Tag, antenna: Antenna, robot_pos: np.ndarray,
                  lambda_signal: float, phase_noise_std: float = 0.09) -> float:
    r = np.linalg.norm(tag.position - robot_pos)

    tx, ty, tz = tag.position
    if tx == 0:
        theta_tag = np.pi / 2
    else:
        theta_tag = np.arctan(ty / tx)
    theta_tag = (theta_tag + 2 * np.pi) % np.pi

    if tz == antenna.z:
        phi_tag = 0.0
    else:
        phi_tag = np.arctan(np.sqrt(tx ** 2 + ty ** 2) / (tz - antenna.z))
    phi_tag = (phi_tag + 2 * np.pi) % np.pi

    phase = (2 * np.pi / lambda_signal) * 2 * r + antenna.phase_bias
    phase += np.random.normal(0, phase_noise_std)
    return phase
