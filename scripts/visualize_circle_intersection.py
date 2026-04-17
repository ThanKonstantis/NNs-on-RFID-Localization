"""Interactive 3-D browser visualisation of the circle-intersection method.

For every tag in the dataset that has >= n_antennas readings, and every
combination of n_antennas readings, the plot shows:
  - All antenna trajectories
  - The predicted circle for each antenna (locus of candidate tag positions)
  - The circle-centre marker (antenna position at the predicted x_tag)
  - The real tag position
  - The predicted 3-D tag position and error line to ground truth

Use the dropdown (top-left) to navigate between tag / antenna combos.

Usage:
    python scripts/visualize_circle_intersection.py
    python scripts/visualize_circle_intersection.py --n-antennas 3
    python scripts/visualize_circle_intersection.py --results-dir results_2D/my_run
    python scripts/visualize_circle_intersection.py --output my_vis.html
"""

import argparse
import sys
import webbrowser
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import plotly.graph_objects as go

from combine_2d_to_3d import (
    find_best_model_dir,
    load_model,
    compute_abs_max,
    preprocess_sample,
    run_inference,
    get_antenna_yz_at_x,
    optimize_3d,
)
from src.preprocessing.dataset import load_experiment_data


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive browser plot of circle-intersection 3-D estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", default="results_2D",
                   help="results_2D folder or a specific run inside it (default: results_2D).")
    p.add_argument("--data", default="Experiments/Experiment_Data.pkl")
    p.add_argument("--n-antennas", type=int, default=2, choices=[2, 3, 4],
                   help="Number of antennas to combine per tag (default: 2).")
    p.add_argument("--output", default=None,
                   help="Output HTML file. Defaults to "
                        "results_3D_combined/circle_vis_<N>ant.html.")
    p.add_argument("--device", default=None)
    return p.parse_args()


# ─── Geometry ─────────────────────────────────────────────────────────────────

def circle_points(center: np.ndarray, radius: float, n: int = 120):
    """Closed loop of 3-D points for a circle with normal=[1,0,0]."""
    theta = np.linspace(0, 2 * np.pi, n + 1)
    x = np.full(n + 1, center[0])
    y = center[1] + radius * np.cos(theta)
    z = center[2] + radius * np.sin(theta)
    return x, y, z


# ─── Colour palette (supports up to 4 antennas) ───────────────────────────────

ANT_LINE_COLORS = ["royalblue",      "seagreen",      "darkorange",   "mediumpurple"]
CIRCLE_COLORS   = ["cornflowerblue", "mediumseagreen", "sandybrown",  "plum"]
CENTER_COLORS   = ["darkblue",       "darkgreen",     "chocolate",    "purple"]


# ─── Traces ───────────────────────────────────────────────────────────────────

def traces_per_combo(n_ant: int) -> int:
    """3 traces per antenna (trajectory + circle + centre) + GT + pred + error line."""
    return 3 * n_ant + 3


def _combo_traces(combo: dict, visible: bool) -> list[go.Scatter3d]:
    """Build Scatter3d traces for one (tag, antenna-combo) entry."""
    traces = []
    lg     = combo["label"]
    n_ant  = len(combo["ant_indices"])

    for k in range(n_ant):
        path   = combo["paths"][k]
        center = combo["centers"][k]
        radius = combo["radii"][k]
        lbl    = f"Antenna {combo['ant_indices'][k]}"

        # Trajectory
        traces.append(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode="lines",
            name=f"{lbl} trajectory",
            line=dict(color=ANT_LINE_COLORS[k], width=4),
            visible=visible,
            legendgroup=lg, showlegend=visible,
        ))

        # Circle
        cx, cy, cz = circle_points(center, radius)
        traces.append(go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="lines",
            name=f"{lbl} circle  r={radius:.1f} cm",
            line=dict(color=CIRCLE_COLORS[k], width=2, dash="dot"),
            visible=visible,
            legendgroup=lg, showlegend=visible,
        ))

        # Circle centre (antenna position at x_pred)
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="markers",
            name=f"{lbl} centre",
            marker=dict(size=7, color=CENTER_COLORS[k], symbol="diamond"),
            visible=visible,
            legendgroup=lg, showlegend=visible,
        ))

    # Ground-truth tag position
    tp = combo["tag_pos"]
    traces.append(go.Scatter3d(
        x=[tp[0]], y=[tp[1]], z=[tp[2]],
        mode="markers",
        name="Real tag",
        marker=dict(size=11, color="red", symbol="cross"),
        visible=visible,
        legendgroup=lg, showlegend=visible,
    ))

    # Predicted 3-D position
    pp  = combo["pred_pos"]
    err = combo["error"]
    traces.append(go.Scatter3d(
        x=[pp[0]], y=[pp[1]], z=[pp[2]],
        mode="markers",
        name=f"Predicted  err={err:.1f} cm",
        marker=dict(size=11, color="orange", symbol="diamond"),
        visible=visible,
        legendgroup=lg, showlegend=visible,
    ))

    # Error line between real and predicted
    traces.append(go.Scatter3d(
        x=[tp[0], pp[0]], y=[tp[1], pp[1]], z=[tp[2], pp[2]],
        mode="lines",
        name=f"Error line  {err:.1f} cm",
        line=dict(color="red", width=2, dash="dash"),
        visible=visible,
        legendgroup=lg, showlegend=visible,
    ))

    expected = traces_per_combo(n_ant)
    assert len(traces) == expected, f"Expected {expected} traces, got {len(traces)}"
    return traces


# ─── Build figure ─────────────────────────────────────────────────────────────

def build_figure(all_combos: list, n_ant: int) -> go.Figure:
    fig  = go.Figure()
    tpc  = traces_per_combo(n_ant)

    for i, combo in enumerate(all_combos):
        for t in _combo_traces(combo, visible=(i == 0)):
            fig.add_trace(t)

    n_total = len(all_combos) * tpc

    buttons = []
    for i, combo in enumerate(all_combos):
        vis   = [False] * n_total
        start = i * tpc
        for j in range(tpc):
            vis[start + j] = True
        buttons.append(dict(
            label=combo["btn_label"],
            method="update",
            args=[
                {"visible": vis},
                {"title": {"text": combo["label"], "font": {"size": 14}}},
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.18, yanchor="top",
            pad=dict(t=5, b=5),
            bgcolor="white",
            bordercolor="#aaa",
            font=dict(size=12),
        )],
        scene=dict(
            xaxis_title="X (cm)",
            yaxis_title="Y (cm)",
            zaxis_title="Z (cm)",
            aspectmode="data",
        ),
        title=dict(
            text=all_combos[0]["label"] if all_combos else "Circle Intersection Visualisation",
            font=dict(size=14),
        ),
        legend=dict(x=1.02, y=1.0, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=0, r=220, t=130, b=0),
        height=820,
    )

    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_ant  = args.n_antennas

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: '{results_dir}' not found.")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: '{data_path}' not found.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        Path(f"results_3D_combined/circle_vis_{n_ant}ant.html")

    # ── Load best 2-D model ───────────────────────────────────────────────────
    best_dir, best_config = find_best_model_dir(results_dir)
    model = load_model(best_dir, best_config, device)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data from {data_path} ...")
    experiment_data = load_experiment_data(data_path)
    print(f"  {len(experiment_data)} tags total")

    # ── Reproduce training-time normalisation ─────────────────────────────────
    print("Computing abs_max from training portion ...")
    abs_max = compute_abs_max(experiment_data)

    # ── Run inference for every (tag, antenna-combo) ──────────────────────────
    all_combos = []
    print(f"\nRunning inference ({n_ant}-antenna combos) ...")

    for tag_idx, tag_readings in enumerate(experiment_data):
        if len(tag_readings) < n_ant:
            continue

        tag_pos = np.array(tag_readings[0]["tag_pos"])

        for ant_indices in combinations(range(len(tag_readings)), n_ant):
            paths   = []
            centers = []
            radii   = []
            circles = []

            for idx in ant_indices:
                reading = tag_readings[idx]
                path    = reading["path"]
                tensor  = preprocess_sample(path, abs_max)
                x_pred, r_pred = run_inference(model, tensor, abs_max, device)
                y_ant, z_ant   = get_antenna_yz_at_x(path, x_pred)
                center = np.array([x_pred, y_ant, z_ant])

                paths.append(path)
                centers.append(center)
                radii.append(r_pred)
                circles.append((center, r_pred))

            pred = optimize_3d(circles)
            err  = float(np.linalg.norm(pred - tag_pos))

            ant_str = ",".join(str(i) for i in ant_indices)
            all_combos.append({
                "label":       (f"Tag {tag_idx}  |  Antennas ({ant_str})  |  "
                                f"Error = {err:.1f} cm"),
                "btn_label":   f"T{tag_idx} A({ant_str})  {err:.0f}cm",
                "ant_indices": list(ant_indices),
                "paths":       paths,
                "centers":     centers,
                "radii":       radii,
                "tag_pos":     tag_pos,
                "pred_pos":    pred,
                "error":       err,
            })

    print(f"  {len(all_combos)} combos across {len(experiment_data)} tags")

    # ── Build Plotly figure ───────────────────────────────────────────────────
    print("\nBuilding interactive figure ...")
    fig = build_figure(all_combos, n_ant)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    abs_out = output_path.resolve()
    print(f"\nSaved → {abs_out}")
    webbrowser.open(f"file://{abs_out}")
    print("Opening in browser ...")


if __name__ == "__main__":
    main()
