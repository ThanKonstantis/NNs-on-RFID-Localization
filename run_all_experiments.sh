#!/usr/bin/env bash
# run_all_experiments.sh
# Run run_experiments.sh for every trajectory × antenna-count combination.
#
# Trajectory rules:
#   straight  (Experiment_Data_Straight.pkl)  →  2, 3, 4 antennas
#   s_path    (Experiment_Data_S.pkl)          →  1, 2, 3, 4 antennas
#   v_path    (Experiment_Data_V.pkl)          →  1, 2, 3, 4 antennas
#
# Note: phase_relock is automatically skipped for 1-antenna runs.
#
# Usage:
#   ./run_all_experiments.sh [<base_name>] [OPTIONS]
#
# Arguments:
#   base_name   Prefix for all run directories (default: YYYYMMDD_HHMMSS).
#               Each run is saved as results/<base_name>_<traj>_<N>ant/.
#
# Options:
#   --data-dir  PATH    Directory containing the pkl files
#                       (default: Experiments)
#   --epochs    N       Passed through to run_experiments.sh (default: 200)
#   --folds     N       Passed through (default: 5)
#   --lr        FLOAT   Passed through (default: 0.01)
#   --batch-size N      Passed through (default: 32)
#   --patience  N       Passed through (default: 90)
#   --device    STR     pytorch device: cpu / cuda (default: auto-detect)

set -uo pipefail   # note: no -e so one failed run doesn't abort the rest

# ─── Locate project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Defaults ───────────────────────────────────────────────────────────────
DATA_DIR="Experiments"
PASSTHROUGH=()
BASE_NAME=""

# ─── Parse arguments ────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \?//' | head -30
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)   DATA_DIR="$2";                          shift 2 ;;
        --epochs)     PASSTHROUGH+=("--epochs"   "$2");       shift 2 ;;
        --folds)      PASSTHROUGH+=("--folds"    "$2");       shift 2 ;;
        --lr)         PASSTHROUGH+=("--lr"       "$2");       shift 2 ;;
        --batch-size) PASSTHROUGH+=("--batch-size" "$2");     shift 2 ;;
        --patience)   PASSTHROUGH+=("--patience" "$2");       shift 2 ;;
        --device)     PASSTHROUGH+=("--device"   "$2");       shift 2 ;;
        -h|--help)    usage ;;
        -*)           echo "Unknown option: $1" >&2; usage ;;
        *)            BASE_NAME="$1";                         shift   ;;
    esac
done

if [[ -z "$BASE_NAME" ]]; then
    BASE_NAME="$(date +%Y%m%d_%H%M%S)"
fi

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python interpreter not found." >&2
    echo "       Set PYTHON env var to the correct path." >&2
    exit 1
fi

# ─── Trajectory definitions ──────────────────────────────────────────────────
# Format: "pkl_filename:ant1 ant2 ..."
TRAJECTORIES=(
    "straight:Experiment_Data_Straight.pkl:2 3 4"
    "s_path:Experiment_Data_S.pkl:1 2 3 4"
    "v_path:Experiment_Data_V.pkl:1 2 3 4"
)

# ─── Run tracking ────────────────────────────────────────────────────────────
SUCCESSFUL_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()

TOTAL_RUNS=0
for ENTRY in "${TRAJECTORIES[@]}"; do
    ANT_STR="${ENTRY##*:}"
    for _ in $ANT_STR; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
done

RUN_IDX=0

# ─── Main loop ───────────────────────────────────────────────────────────────
for ENTRY in "${TRAJECTORIES[@]}"; do
    TRAJ="${ENTRY%%:*}"
    REST="${ENTRY#*:}"
    PKL_FILE="$DATA_DIR/${REST%%:*}"
    ANT_LIST="${REST##*:}"

    if [[ ! -f "$PKL_FILE" ]]; then
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "  WARNING: '$PKL_FILE' not found — skipping $TRAJ entirely"
        echo "╚══════════════════════════════════════════════════════════╝"
        for N_ANT in $ANT_LIST; do
            SKIPPED_RUNS+=("${BASE_NAME}_${TRAJ}_${N_ANT}ant")
        done
        continue
    fi

    for N_ANT in $ANT_LIST; do
        RUN_IDX=$((RUN_IDX + 1))
        RUN_NAME="${BASE_NAME}_${TRAJ}_${N_ANT}ant"

        echo ""
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "  [$RUN_IDX/$TOTAL_RUNS]  $TRAJ  |  $N_ANT antenna(s)"
        echo "  Run name : $RUN_NAME"
        echo "  Data     : $PKL_FILE"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""

        ./run_experiments.sh "$RUN_NAME" \
            --antennas "$N_ANT" \
            --data     "$PKL_FILE" \
            "${PASSTHROUGH[@]}" \
            && SUCCESSFUL_RUNS+=("$RUN_NAME") \
            || FAILED_RUNS+=("$RUN_NAME")

        echo ""
    done
done

# ─── Final summary ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  All experiments complete"
echo "  Base name  : $BASE_NAME"
echo "  Successful : ${#SUCCESSFUL_RUNS[@]}"
for r in "${SUCCESSFUL_RUNS[@]}"; do echo "    ✓  $r"; done
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
echo "  Failed     : ${#FAILED_RUNS[@]}"
for r in "${FAILED_RUNS[@]}"; do echo "    ✗  $r"; done
fi
if [[ ${#SKIPPED_RUNS[@]} -gt 0 ]]; then
echo "  Skipped    : ${#SKIPPED_RUNS[@]}"
for r in "${SKIPPED_RUNS[@]}"; do echo "    -  $r"; done
fi
echo "╚══════════════════════════════════════════════════════════╝"
