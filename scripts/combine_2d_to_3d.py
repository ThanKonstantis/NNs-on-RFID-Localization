"""Combine 2-D model predictions from multiple antenna readings to estimate
the 3-D tag position via circle intersection.

For each tag in the dataset that has >= n_antennas readings, every combination
of n_antennas readings is processed:

  1. Each reading is passed through the best 2-D model → (x_pred, r_pred)
     where r_pred is the radius of the circular locus in 3-D space.
  2. The antenna's (y, z) at x = x_pred is interpolated from the raw path.
  3. A circle is constructed: centre A=(x_pred, y_ant, z_ant),
     radius R=r_pred, normal N=[1, 0, 0] (straight aperture along X).
  4. The 3-D tag position is estimated by minimising the sum of squared
     distances from a candidate point P to all circles:

         A3D_tag = argmin_P  Σᵢ D(P, Cᵢ)²

     where  D(P, C) = sqrt( (N·(P-A))²  +  (||N×(P-A)|| - R)² )

  5. The estimate is compared to the ground-truth [x, y, z] tag position.

The script scans the results_2D/ tree at any depth for metrics.json files
and picks the experiment with the lowest mean_distance_error_cm that also
has a model.pth.

Usage:
    python scripts/combine_2d_to_3d.py
    python scripts/combine_2d_to_3d.py --results-dir results_2D
    python scripts/combine_2d_to_3d.py --results-dir results_2D/my_run --n-antennas 3
    python scripts/combine_2d_to_3d.py --results-dir results_2D \\
        --data Experiments/Experiment_Data.pkl --output-dir results_3D_combined/my_run/2ant
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from sklearn.model_selection import train_test_split

from src.models.networks import MODEL_REGISTRY
from src.preprocessing.dataset import load_experiment_data, build_2d_arrays


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine 2-D model predictions into 3-D positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results_2D",
                        help="Path to a results_2D/ folder or a specific run inside it "
                             "(default: results_2D).")
    parser.add_argument("--data", default="Experiments/Experiment_Data.pkl",
                        help="Path to Experiment_Data.pkl.")
    parser.add_argument("--n-antennas", type=int, default=2, choices=[2, 3, 4],
                        help="Number of antenna readings to combine per tag (default: 2).")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save results. "
                             "Defaults to results_3D_combined/<results_dir_name>/<N>ant/.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# ─── Model loading ────────────────────────────────────────────────────────────

def find_best_model_dir(results_dir: Path) -> tuple[Path, dict]:
    """Recursively scan results_dir for metrics.json and return the experiment
    directory with the lowest mean_distance_error_cm that also has a model.pth."""
    best_dir, best_config = None, None
    best_error = float("inf")

    for metrics_file in sorted(results_dir.rglob("metrics/metrics.json")):
        exp_dir = metrics_file.parent.parent   # .../exp_key/metrics/metrics.json
        if not (exp_dir / "model.pth").exists():
            continue                            # skip baselines (no weights)
        with open(metrics_file) as f:
            m = json.load(f)
        err = m.get("mean_distance_error_cm", float("inf"))
        if err < best_error:
            best_error = err
            best_dir   = exp_dir
            best_config = m

    if best_dir is None:
        raise FileNotFoundError(
            f"No experiment with model.pth + metrics.json found under {results_dir}"
        )

    print(f"Best model: {best_dir.relative_to(results_dir)}  "
          f"(mean error {best_error:.2f} cm)")
    return best_dir, best_config


def _parse_hidden(hidden_str):
    return tuple(int(x) for x in hidden_str.split(","))


def load_model(model_dir: Path, config: dict, device: str) -> nn.Module:
    """Reconstruct the model from saved config and load its weights."""
    import inspect

    run_cfg    = config.get("config", {})
    model_name = run_cfg.get("model", "leaky_relu")
    hidden_str = run_cfg.get("hidden", "default")

    model_cls = MODEL_REGISTRY[model_name]

    # Derive input_size from the first weight tensor in the state dict
    pth_path   = model_dir / "model.pth"
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    input_size = None
    for key, val in state_dict.items():
        if "weight" in key and val.dim() == 2:
            input_size = val.shape[1]
            break
    if input_size is None:
        raise ValueError("Could not infer input_size from state dict.")

    model_params = {"input_size": input_size, "output_size": 2}

    if hidden_str != "default":
        hidden = _parse_hidden(hidden_str)
        sig = inspect.signature(model_cls.__init__)
        if "hidden_units" in sig.parameters:
            default_val = sig.parameters["hidden_units"].default
            if isinstance(default_val, (list, tuple)):
                model_params["hidden_units"] = hidden
            else:
                model_params["hidden_units"] = hidden[0]

    model = model_cls(**model_params)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded {model_cls.__name__}  input_size={input_size}  from {pth_path}")
    return model


# ─── Normalisation ───────────────────────────────────────────────────────────

def compute_abs_max(experiment_data: list) -> np.ndarray:
    """Reproduce the training-time abs_max by mirroring cross_validate_2d's split."""
    info_tensor, rfid_label = build_2d_arrays(experiment_data)

    # Same split as cross_validate_2d
    X_main, _, _, _ = train_test_split(
        info_tensor, rfid_label, test_size=0.10, random_state=42
    )

    # Phase offset removal (matches make_eval_dataloaders_2d)
    X = X_main.copy()
    for i in range(len(X)):
        X[i, :, -1] -= np.min(X[i, :, -1])

    abs_max = np.abs(X).max(axis=(0, 1))
    abs_max[1] = abs_max[0]          # Y shares X scale
    abs_max[abs_max == 0] = 1.0
    return abs_max


# ─── Single-sample inference ─────────────────────────────────────────────────

def preprocess_sample(path_full: np.ndarray, abs_max: np.ndarray) -> torch.Tensor:
    """Preprocess one raw antenna path for model inference.

    path_full : (seq_len, 4)  [x_ant, y_ant, z_ant, phase]
    Returns   : float32 tensor of shape (1, seq_len*3)
    """
    sample = path_full[:, [0, 1, 3]].copy()    # drop z_ant → [x, y, phase]
    sample[:, -1] -= np.min(sample[:, -1])      # phase offset removal
    sample_norm = sample / abs_max              # normalise
    return torch.tensor(sample_norm.reshape(1, -1), dtype=torch.float32)


def run_inference(model, tensor: torch.Tensor, abs_max: np.ndarray,
                  device: str) -> tuple[float, float]:
    """Run model and denormalise output → (x_pred_cm, r_pred_cm)."""
    with torch.inference_mode():
        pred = model(tensor.to(device)).cpu().numpy()[0]  # [x_norm, r_norm]
    x_pred = float(pred[0]) * abs_max[0]
    r_pred = float(pred[1]) * abs_max[1]   # abs_max[1] == abs_max[0] (shared scale)
    return x_pred, r_pred


# ─── Circle geometry ─────────────────────────────────────────────────────────

def get_antenna_yz_at_x(path_full: np.ndarray, x_target: float) -> tuple[float, float]:
    """Interpolate antenna y and z from the full 4-col path at x = x_target.

    path_full : (seq_len, 4)  [x_ant, y_ant, z_ant, phase]
    """
    x_ant = path_full[:, 0]
    y_ant = path_full[:, 1]
    z_ant = path_full[:, 2]

    sort_idx = np.argsort(x_ant)
    xs = x_ant[sort_idx]
    ys = y_ant[sort_idx]
    zs = z_ant[sort_idx]

    y_interp = float(np.interp(x_target, xs, ys))
    z_interp = float(np.interp(x_target, xs, zs))
    return y_interp, z_interp


def dist_to_circle(P: np.ndarray, center: np.ndarray, radius: float) -> float:
    """Distance from point P to circle (centre, radius, normal=[1,0,0]).

    D = sqrt( (P[0]-A[0])²  +  (sqrt((P[1]-A[1])²+(P[2]-A[2])²) - R)² )
    """
    along = P[0] - center[0]
    perp  = np.sqrt((P[1] - center[1]) ** 2 + (P[2] - center[2]) ** 2)
    return np.sqrt(along ** 2 + (perp - radius) ** 2)


def _residuals(P, circles):
    return [dist_to_circle(P, A, R) for A, R in circles]


def optimize_3d(circles: list) -> np.ndarray:
    """Find 3-D point minimising Σ D(P, Cᵢ)².

    Tags are always in the positive-Y half-space, so Y is bounded to [0, ∞)
    and the initial guess is forced into positive Y to avoid the mirror solution.

    circles : list of (center_array, radius) tuples
    Returns : estimated tag position [px, py, pz]
    """
    centers = np.array([A for A, _ in circles])
    x0 = centers.mean(axis=0)
    x0[1] = abs(x0[1])          # start in positive-Y half-space

    result = least_squares(
        _residuals,
        x0=x0,
        args=(circles,),
        method="trf",
        bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
    )
    return result.x


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_predictions_3d(y_true: np.ndarray, y_pred: np.ndarray,
                        distances: np.ndarray, mean_err: float, std_err: float,
                        n: int = 20, save_path: Path = None) -> None:
    """3-D scatter of ground truth vs predicted positions (first n combos)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = min(n, len(y_true))
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*y_true[:n].T, color="blue", label="Ground Truth", s=80)
    ax.scatter(*y_pred[:n].T, color="red",  label="Predicted",    s=80, marker="^")

    for gt, pred, d in zip(y_true[:n], y_pred[:n], distances[:n]):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], [gt[2], pred[2]],
                color="gray", linestyle="--", linewidth=0.8)
        mid = (gt + pred) / 2
        ax.text(*mid, f"{d:.1f}", fontsize=7, ha="center",
                bbox=dict(facecolor="white", alpha=0.3, edgecolor="none"))

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title(
        f"Ground Truth vs Predicted (3-D)\n"
        f"Mean: {mean_err:.2f} cm   Std: {std_err:.2f} cm"
    )
    ax.legend()
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  predictions_3d.png")
    else:
        plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: '{results_dir}' not found.")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: '{data_path}' not found.")
        sys.exit(1)

    n_ant = args.n_antennas

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results_3D_combined") / results_dir.name / f"{n_ant}ant"

    metrics_dir = output_dir / "metrics"
    images_dir  = output_dir / "images"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── Load best model ───────────────────────────────────────────────────────
    best_dir, best_config = find_best_model_dir(results_dir)
    model = load_model(best_dir, best_config, device)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data from {data_path} ...")
    experiment_data = load_experiment_data(data_path)
    print(f"  {len(experiment_data)} tags total")

    # ── Reproduce training-time normalisation ─────────────────────────────────
    print("Computing abs_max from training portion ...")
    abs_max = compute_abs_max(experiment_data)

    # ── Combination loop ──────────────────────────────────────────────────────
    errors   = []
    gt_list  = []
    pred_list = []
    n_tags   = 0
    n_combos = 0

    print(f"\nCombining {n_ant}-antenna predictions ...")

    for tag_readings in experiment_data:
        if len(tag_readings) < n_ant:
            continue

        n_tags += 1
        tag_pos = np.array(tag_readings[0]["tag_pos"])   # ground truth [x, y, z]

        for combo in combinations(tag_readings, n_ant):
            circles = []
            for reading in combo:
                path_full = reading["path"]              # (seq_len, 4)

                tensor = preprocess_sample(path_full, abs_max)
                x_pred, r_pred = run_inference(model, tensor, abs_max, device)

                y_ant, z_ant = get_antenna_yz_at_x(path_full, x_pred)
                center = np.array([x_pred, y_ant, z_ant])
                circles.append((center, r_pred))

            p3d = optimize_3d(circles)
            err = float(np.linalg.norm(p3d - tag_pos))
            errors.append(err)
            gt_list.append(tag_pos)
            pred_list.append(p3d)
            n_combos += 1

    # ── Metrics ───────────────────────────────────────────────────────────────
    errors   = np.array(errors)
    y_true   = np.array(gt_list)
    y_pred   = np.array(pred_list)
    mean_err = float(np.mean(errors))
    std_err  = float(np.std(errors))
    pcts     = np.percentile(errors, [25, 50, 75, 90, 95, 99])

    print(f"\n{'─'*50}")
    print(f"  Tags with >= {n_ant} antennas : {n_tags}")
    print(f"  Combinations evaluated      : {n_combos}")
    print(f"  Mean 3-D error              : {mean_err:.2f} cm")
    print(f"  Std                         : {std_err:.2f} cm")
    print(f"  p50                         : {pcts[1]:.2f} cm")
    print(f"  p90                         : {pcts[3]:.2f} cm")
    print(f"{'─'*50}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    metrics = {
        "source_model":            str(best_dir.relative_to(results_dir)),
        "source_mean_error_2d_cm": best_config.get("mean_distance_error_cm"),
        "n_antennas_combined":     n_ant,
        "n_tags":                  n_tags,
        "n_combinations":          n_combos,
        "mean_distance_error_cm":  round(mean_err, 4),
        "std_cm":                  round(std_err, 4),
        "percentiles_cm": {
            "p25": round(float(pcts[0]), 4),
            "p50": round(float(pcts[1]), 4),
            "p75": round(float(pcts[2]), 4),
            "p90": round(float(pcts[3]), 4),
            "p95": round(float(pcts[4]), 4),
            "p99": round(float(pcts[5]), 4),
        },
    }

    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(metrics_dir / "distances.npy", errors)

    plot_predictions_3d(y_true, y_pred, errors, mean_err, std_err,
                        save_path=images_dir / "predictions_3d.png")

    print(f"\nSaved → {output_dir}/")
    print(f"  metrics/metrics.json")
    print(f"  metrics/distances.npy")


if __name__ == "__main__":
    main()
