"""Interactive 3-D browser visualisation of the circle-intersection method.

For every tag in the dataset that has >= 2 antenna readings, and every
combination of 2 antenna readings, the plot shows:
  - Both antenna trajectories
  - The predicted circle for each antenna (locus of candidate tag positions)
  - The circle-centre marker (antenna position at the predicted x_tag)
  - The real tag position
  - The predicted 3-D tag position and error line to ground truth

Use the dropdown (top-left) to navigate between tag / antenna-pair combos.

Usage:
    python scripts/visualize_circle_intersection.py
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
    p.add_argument("--output", default="results_3D_combined/circle_vis.html",
                   help="Output HTML file (default: results_3D_combined/circle_vis.html).")
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


# ─── Plotly traces ────────────────────────────────────────────────────────────

ANT_LINE_COLORS   = ["royalblue",       "seagreen"]
CIRCLE_COLORS     = ["cornflowerblue",  "mediumseagreen"]
CENTER_COLORS     = ["darkblue",        "darkgreen"]

# Fixed number of traces emitted per combo — must not change.
TRACES_PER_COMBO  = 9   # 2 traj + 2 circles + 2 centres + GT + pred + error line


def _combo_traces(combo: dict, visible: bool) -> list[go.Scatter3d]:
    """Build the 9 Scatter3d traces for one (tag, antenna-pair) combo."""
    traces = []
    lg     = combo["label"]          # shared legendgroup keeps legend tidy

    ant_labels = [f"Antenna {combo['ant_i']}", f"Antenna {combo['ant_j']}"]

    for k, (path, center, radius) in enumerate([
        (combo["path1"], combo["center1"], combo["radius1"]),
        (combo["path2"], combo["center2"], combo["radius2"]),
    ]):
        lbl = ant_labels[k]

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

    assert len(traces) == TRACES_PER_COMBO, f"Expected {TRACES_PER_COMBO}, got {len(traces)}"
    return traces


# ─── Build figure ─────────────────────────────────────────────────────────────

def build_figure(all_combos: list) -> go.Figure:
    fig = go.Figure()

    for i, combo in enumerate(all_combos):
        for t in _combo_traces(combo, visible=(i == 0)):
            fig.add_trace(t)

    n_total = len(all_combos) * TRACES_PER_COMBO

    buttons = []
    for i, combo in enumerate(all_combos):
        vis      = [False] * n_total
        start    = i * TRACES_PER_COMBO
        for j in range(TRACES_PER_COMBO):
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

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: '{results_dir}' not found.")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: '{data_path}' not found.")
        sys.exit(1)

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

    # ── Run inference for every (tag, antenna-pair) combo ─────────────────────
    all_combos = []
    print("\nRunning inference ...")

    for tag_idx, tag_readings in enumerate(experiment_data):
        if len(tag_readings) < 2:
            continue

        tag_pos = np.array(tag_readings[0]["tag_pos"])

        for ant_i, ant_j in combinations(range(len(tag_readings)), 2):
            r1 = tag_readings[ant_i]
            r2 = tag_readings[ant_j]

            path1 = r1["path"]
            t1    = preprocess_sample(path1, abs_max)
            x1, rad1 = run_inference(model, t1, abs_max, device)
            y1, z1   = get_antenna_yz_at_x(path1, x1)
            center1  = np.array([x1, y1, z1])

            path2 = r2["path"]
            t2    = preprocess_sample(path2, abs_max)
            x2, rad2 = run_inference(model, t2, abs_max, device)
            y2, z2   = get_antenna_yz_at_x(path2, x2)
            center2  = np.array([x2, y2, z2])

            pred = optimize_3d([(center1, rad1), (center2, rad2)])
            err  = float(np.linalg.norm(pred - tag_pos))

            all_combos.append({
                "label":    (f"Tag {tag_idx}  |  Antennas ({ant_i}, {ant_j})  |  "
                             f"Error = {err:.1f} cm"),
                "btn_label": f"T{tag_idx} A({ant_i},{ant_j})  {err:.0f} cm",
                "ant_i":    ant_i,
                "ant_j":    ant_j,
                "path1":    path1,
                "path2":    path2,
                "center1":  center1,
                "radius1":  rad1,
                "center2":  center2,
                "radius2":  rad2,
                "tag_pos":  tag_pos,
                "pred_pos": pred,
                "error":    err,
            })

    print(f"  {len(all_combos)} combos across {len(experiment_data)} tags")

    # ── Build Plotly figure ───────────────────────────────────────────────────
    print("\nBuilding interactive figure ...")
    fig = build_figure(all_combos)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    abs_out = output_path.resolve()
    print(f"\nSaved → {abs_out}")
    webbrowser.open(f"file://{abs_out}")
    print("Opening in browser ...")


if __name__ == "__main__":
    main()
