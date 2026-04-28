#!/usr/bin/env python3
"""
collect_inference_times.py
---------------------------
Aggregate inference timing data produced by run_all_experiments.sh.

For every model found across all <base>_*ant result directories, produce a
CSV with one row per (path, n_antennas) combination and save it inside a
per-model folder:

    results/<base>_inference_times/<model>/inference_times.csv

Columns:
    path, n_antennas, evaluation_time_s, n_samples, mean_inference_ms

Usage:
    python scripts/collect_inference_times.py <base_name>
    python scripts/collect_inference_times.py 20260417_185554
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


def parse_run_dir(name: str, base: str):
    """
    Parse trajectory and n_antennas from a run directory name.
    Expected format: {base}_{trajectory}_{N}ant
    Returns (trajectory, n_antennas) or None if not matching.
    """
    prefix = base + "_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix):]  # e.g. "s_path_3ant"
    m = re.match(r"^(.+?)_(\d+)ant$", remainder)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-model inference times across all path/antenna combinations."
    )
    parser.add_argument(
        "base_name",
        help="Base name used when running run_all_experiments.sh (e.g. 20260417_185554)",
    )
    args = parser.parse_args()

    base = args.base_name
    results_root = Path("results")

    if not results_root.exists():
        print(f"ERROR: '{results_root}' directory not found.", file=sys.stderr)
        sys.exit(1)

    # model_name -> list of (trajectory, n_antennas, evaluation_time_s, n_samples)
    model_data: dict[str, list] = {}

    run_dirs = sorted(d for d in results_root.iterdir() if d.is_dir())
    matched = 0

    for run_dir in run_dirs:
        parsed = parse_run_dir(run_dir.name, base)
        if parsed is None:
            continue
        trajectory, n_antennas = parsed
        matched += 1

        for model_dir in sorted(d for d in run_dir.iterdir() if d.is_dir()):
            model_name = model_dir.name
            metrics_path = model_dir / "metrics" / "metrics.json"
            distances_path = model_dir / "metrics" / "distances.npy"

            if not metrics_path.exists():
                continue

            with open(metrics_path) as f:
                metrics = json.load(f)

            evaluation_time_s = metrics.get("evaluation_time_s")
            if evaluation_time_s is None:
                print(
                    f"  WARNING: no evaluation_time_s in {metrics_path}, skipping.",
                    file=sys.stderr,
                )
                continue

            n_samples = None
            if distances_path.exists():
                n_samples = len(np.load(distances_path))

            model_data.setdefault(model_name, []).append(
                (trajectory, n_antennas, evaluation_time_s, n_samples)
            )

    if matched == 0:
        print(
            f"ERROR: no run directories found matching base '{base}' under '{results_root}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not model_data:
        print(
            "ERROR: run directories were found but no metrics.json files could be read.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Write one CSV per model inside results/<base>_inference_times/<model>/
    out_root = results_root / f"{base}_inference_times"
    out_root.mkdir(parents=True, exist_ok=True)

    FIELDNAMES = ["path", "n_antennas", "evaluation_time_s", "n_samples", "mean_inference_ms"]

    for model_name, rows in sorted(model_data.items()):
        model_dir = out_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        csv_path = model_dir / "inference_times.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for trajectory, n_antennas, eval_time, n_samples in sorted(
                rows, key=lambda r: (r[0], r[1])
            ):
                mean_ms = (
                    round(eval_time * 1000 / n_samples, 4)
                    if (n_samples and n_samples > 0)
                    else ""
                )
                writer.writerow(
                    {
                        "path": trajectory,
                        "n_antennas": n_antennas,
                        "evaluation_time_s": round(eval_time, 6),
                        "n_samples": n_samples if n_samples is not None else "",
                        "mean_inference_ms": mean_ms,
                    }
                )

        print(f"  {csv_path}")

    print(f"\nDone. {len(model_data)} model(s) written to: {out_root}/")


if __name__ == "__main__":
    main()
