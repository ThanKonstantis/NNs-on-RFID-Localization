#!/usr/bin/env bash
# build_data_2d.sh
# Build final_tensor.npy and final_labels.npy for 2-D (single-antenna) experiments.
#
# Usage:
#   ./build_data_2d.sh [OPTIONS]
#
# Options:
#   --measurements-dir PATH   Path to the Measurements folder
#                             (default: Experiments/Measurements)
#   --experiment       STR    Substring to filter experiment folders
#                             (default: "Straight ")
#   --interp-length    N      Number of interpolation points per trajectory
#                             (default: 385)
#   --size-threshold   N      Minimum file size in bytes to include a reading
#                             (default: 10240)
#   --output-dir       PATH   Output directory for the .npy files
#                             (default: auto-increments Raw_Data_Single_Antenna_<N>
#                              inside Experiments/)

set -euo pipefail

# ─── Locate project root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Defaults ────────────────────────────────────────────────────────────────
MEASUREMENTS_DIR="Experiments/Measurements"
EXPERIMENT="Straight "
INTERP_LENGTH=385
SIZE_THRESHOLD=10240
OUTPUT_DIR=""

# ─── Usage ───────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 1
}

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --measurements-dir) MEASUREMENTS_DIR="$2"; shift 2 ;;
        --experiment)       EXPERIMENT="$2";       shift 2 ;;
        --interp-length)    INTERP_LENGTH="$2";    shift 2 ;;
        --size-threshold)   SIZE_THRESHOLD="$2";   shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";       shift 2 ;;
        -h|--help)          usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ─── Validate environment ─────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python interpreter not found." >&2
    echo "       Set the PYTHON environment variable to the correct path." >&2
    exit 1
fi

if [[ ! -d "$MEASUREMENTS_DIR" ]]; then
    echo "ERROR: Measurements directory '$MEASUREMENTS_DIR' not found." >&2
    exit 1
fi

# ─── Print banner ─────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "  Building 2D single-antenna dataset"
echo "  Date              : $(date)"
echo "  Measurements dir  : $MEASUREMENTS_DIR"
echo "  Experiment filter : $EXPERIMENT"
echo "  Interp length     : $INTERP_LENGTH"
echo "  Size threshold    : $SIZE_THRESHOLD bytes"
echo "  Output dir        : ${OUTPUT_DIR:-auto}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Build argument list ──────────────────────────────────────────────────────
ARGS=(
    --measurements-dir "$MEASUREMENTS_DIR"
    --experiment       "$EXPERIMENT"
    --interp-length    "$INTERP_LENGTH"
    --size-threshold   "$SIZE_THRESHOLD"
)

if [[ -n "$OUTPUT_DIR" ]]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi

# ─── Run ──────────────────────────────────────────────────────────────────────
"$PYTHON" scripts/build_data_2d.py "${ARGS[@]}"
