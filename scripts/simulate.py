"""Generate synthetic RFID simulation data.

Usage:
    python scripts/simulate.py
    python scripts/simulate.py --antennas 3 --y-samples 50 --z-samples 50 --output Simulation/sim_out.pkl
    python scripts/simulate.py --pattern Simulation/pattern.xlsx --antennas 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.simulation.data_generation import interpolate_data, simulate_straight_path_3d

DEFAULT_HEIGHTS = {
    1: [0.633],
    2: [0.633, 1.073],
    3: [0.633, 1.073, 1.528],
    4: [0.633, 1.073, 1.528, 1.928],
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RFID simulation data.")
    parser.add_argument("--pattern", default="Simulation/pattern.xlsx",
                        help="Path to antenna gain pattern Excel file")
    parser.add_argument("--antennas", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of antennas (default: 4)")
    parser.add_argument("--antenna-heights", nargs="+", type=float, default=None,
                        help="Antenna heights in meters (overrides --antennas default heights)")
    parser.add_argument("--x-tag", type=float, default=2.0,
                        help="Tag x position (default: 2.0)")
    parser.add_argument("--y-min", type=float, default=0.1)
    parser.add_argument("--y-max", type=float, default=3.0)
    parser.add_argument("--y-samples", type=int, default=30)
    parser.add_argument("--z-min", type=float, default=0.1)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--z-samples", type=int, default=30)
    parser.add_argument("--path-length", type=float, default=4.0,
                        help="Robot path length in meters (default: 4.0)")
    parser.add_argument("--interp-length", type=int, default=385,
                        help="Interpolation length (default: 385)")
    parser.add_argument("--output", default=None,
                        help="Output pickle file path (default: Simulation/Sim_Data_<N>.pkl)")
    args = parser.parse_args()

    heights = args.antenna_heights or DEFAULT_HEIGHTS[args.antennas]

    if args.output is None:
        out_dir = Path("Simulation")
        # Auto-increment output filename
        existing = [int(p.stem.split("_")[-1]) for p in out_dir.glob("Sim_Data_*.pkl")
                    if p.stem.split("_")[-1].isdigit()]
        idx = max(existing) + 1 if existing else 0
        args.output = str(out_dir / f"Sim_Data_{idx}.pkl")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    y_steps = np.linspace(args.y_min, args.y_max, args.y_samples)
    z_steps = np.linspace(args.z_min, args.z_max, args.z_samples)

    print(f"Simulating {args.y_samples * args.z_samples} tag positions "
          f"with {len(heights)} antenna(s) at heights {heights} m ...")

    data = simulate_straight_path_3d(
        pattern_path=args.pattern,
        antenna_heights=heights,
        path_length=args.path_length,
        x_tag=args.x_tag,
        y_steps=y_steps,
        z_steps=z_steps,
        output_path=None,
    )

    print(f"Interpolating to {args.interp_length} samples per path ...")
    data = interpolate_data(data, length=args.interp_length)

    import pickle
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} samples → {args.output}")


if __name__ == "__main__":
    main()
