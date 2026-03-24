"""Generate visualizations from experiment data or saved results.

Usage:
    # CDF plot from a saved distances file
    python scripts/visualize.py --mode cdf --distances results/distances.npy

    # Quick data overview from the experiment pickle
    python scripts/visualize.py --mode data --data Experiments/Experiment_Data.pkl

    # 3-D scatter of tag positions
    python scripts/visualize.py --mode positions --data Experiments/Experiment_Data.pkl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_cdf(distances, label="Model", save_path=None):
    sorted_d = np.sort(distances)
    cdf = np.arange(len(sorted_d)) / (len(sorted_d) - 1)
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_d, cdf, lw=2, label=label, color="darkblue")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel("Distance Error (cm)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Distance Errors")
    plt.xlim(left=0)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved → {save_path}")
    plt.show()


def plot_tag_positions(data, save_path=None):
    from mpl_toolkits.mplot3d import Axes3D
    positions = [entry[0]["tag_pos"] for entry in data]
    positions = np.array(positions)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=20)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title(f"Tag Positions ({len(positions)} tags)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved → {save_path}")
    plt.show()


def plot_phase_example(data, tag_idx=0):
    """Plot phase vs distance for one tag across all antennas."""
    entry = data[tag_idx]
    plt.figure(figsize=(10, 5))
    for ant_data in entry:
        path = np.array(ant_data["path"])
        plt.plot(path[:, 0], path[:, -1], label=f"Antenna {ant_data['Antenna']}")
    pos = entry[0]["tag_pos"]
    plt.title(f"Phase vs X  |  Tag @ ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) cm")
    plt.xlabel("X position (cm)")
    plt.ylabel("Unwrapped phase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment data or results.")
    parser.add_argument("--mode", choices=["cdf", "data", "positions", "phase"],
                        default="positions", help="Visualization mode")
    parser.add_argument("--data", default="Experiments/Experiment_Data.pkl",
                        help="Path to Experiment_Data.pkl")
    parser.add_argument("--distances", help="Path to .npy file with distance errors (for --mode cdf)")
    parser.add_argument("--tag-idx", type=int, default=0,
                        help="Tag index for phase plot (default: 0)")
    parser.add_argument("--save", help="Save figure to this path instead of showing")
    args = parser.parse_args()

    if args.mode == "cdf":
        if not args.distances:
            print("--distances is required for cdf mode")
            sys.exit(1)
        distances = np.load(args.distances)
        plot_cdf(distances, save_path=args.save)

    else:
        print(f"Loading {args.data} ...")
        with open(args.data, "rb") as f:
            data = pickle.load(f)
        print(f"  {len(data)} tag entries loaded")

        if args.mode == "positions":
            plot_tag_positions(data, save_path=args.save)
        elif args.mode == "phase":
            plot_phase_example(data, tag_idx=args.tag_idx)
        elif args.mode == "data":
            plot_tag_positions(data)
            plot_phase_example(data, tag_idx=args.tag_idx)


if __name__ == "__main__":
    main()
