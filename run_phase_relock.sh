#!/usr/bin/env bash
# run_phase_relock.sh
# Run the Phase Relock baseline for every trajectory × antenna-count combination.
#
# Mirrors the trajectory/antenna rules from run_all_experiments.sh:
#   straight  (Experiment_Data_Straight.pkl)  →  2, 3, 4 antennas
#   s_path    (Experiment_Data_S.pkl)          →  1, 2, 3, 4 antennas
#   v_path    (Experiment_Data_V.pkl)          →  1, 2, 3, 4 antennas
#
# Usage:
#   ./run_phase_relock.sh [<base_name>] [OPTIONS]
#
# Arguments:
#   base_name   Prefix for the results directory (default: YYYYMMDD_HHMMSS).
#               Results are saved under results/<base_name>_phase_relock/.
#
# Options:
#   --data-dir  PATH    Directory containing the pkl files (default: Experiments)
#   --holdout   FLOAT   Holdout fraction passed to run_baseline.py (default: 0.1)

set -uo pipefail   # no -e so one failed run doesn't abort the rest

# ─── Locate project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Defaults ───────────────────────────────────────────────────────────────
DATA_DIR="Experiments"
HOLDOUT="0.1"
BASE_NAME=""

# ─── Parse arguments ────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \?//' | head -20
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2";  shift 2 ;;
        --holdout)  HOLDOUT="$2";   shift 2 ;;
        -h|--help)  usage ;;
        -*)         echo "Unknown option: $1" >&2; usage ;;
        *)          BASE_NAME="$1"; shift ;;
    esac
done

if [[ -z "$BASE_NAME" ]]; then
    BASE_NAME="$(date +%Y%m%d_%H%M%S)"
fi

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python interpreter not found." >&2
    echo "       Set the PYTHON environment variable to the correct path." >&2
    exit 1
fi

# ─── Trajectory definitions ──────────────────────────────────────────────────
# Format: "label:pkl_filename:ant1 ant2 ..."
TRAJECTORIES=(
    "straight:Experiment_Data_Straight.pkl:2 3 4"
    "s_path:Experiment_Data_S.pkl:1 2 3 4"
    "v_path:Experiment_Data_V.pkl:1 2 3 4"
)

# ─── Setup output directories ────────────────────────────────────────────────
RESULTS_DIR="results/${BASE_NAME}_phase_relock"

if [[ -d "$RESULTS_DIR" ]]; then
    echo "ERROR: '$RESULTS_DIR' already exists." >&2
    echo "       Choose a different base name or delete the existing directory." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
export MPLBACKEND=Agg

LOG_FILE="$RESULTS_DIR/run.log"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
ERRORS_LOG="$RESULTS_DIR/errors.log"

echo "trajectory,antennas,mean_error_cm,std_cm,p25_cm,p50_cm,p75_cm,p90_cm,p95_cm,p99_cm" \
    > "$SUMMARY_CSV"

# ─── Count total runs ────────────────────────────────────────────────────────
TOTAL_RUNS=0
for ENTRY in "${TRAJECTORIES[@]}"; do
    ANT_STR="${ENTRY##*:}"
    for _ in $ANT_STR; do
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done
done

# ─── Banner ──────────────────────────────────────────────────────────────────
{
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "  Phase Relock baseline"
    echo "  Base name : $BASE_NAME"
    echo "  Date      : $(date)"
    echo "  Data dir  : $DATA_DIR"
    echo "  Holdout   : $HOLDOUT"
    echo "  Results   : $RESULTS_DIR"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
} | tee "$LOG_FILE"

# ─── Run tracking ────────────────────────────────────────────────────────────
SUCCESSFUL_RUNS=()
FAILED_RUNS=()
SKIPPED_RUNS=()
RUN_IDX=0

# ─── Main loop ───────────────────────────────────────────────────────────────
for ENTRY in "${TRAJECTORIES[@]}"; do
    TRAJ="${ENTRY%%:*}"
    REST="${ENTRY#*:}"
    PKL_FILE="$DATA_DIR/${REST%%:*}"
    ANT_LIST="${REST##*:}"

    if [[ ! -f "$PKL_FILE" ]]; then
        echo "╔══════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
        echo "  WARNING: '$PKL_FILE' not found — skipping $TRAJ entirely" | tee -a "$LOG_FILE"
        echo "╚══════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
        for N_ANT in $ANT_LIST; do
            SKIPPED_RUNS+=("${TRAJ}_${N_ANT}ant")
        done
        continue
    fi

    for N_ANT in $ANT_LIST; do
        RUN_IDX=$((RUN_IDX + 1))
        EXP_KEY="${TRAJ}_${N_ANT}ant"
        EXP_DIR="$RESULTS_DIR/$EXP_KEY"
        mkdir -p "$EXP_DIR/metrics" "$EXP_DIR/images"

        {
            echo "──────────────────────────────────────────────────────────"
            echo "  [$RUN_IDX/$TOTAL_RUNS]  $TRAJ  |  $N_ANT antenna(s)"
            echo "  Data : $PKL_FILE"
            echo "──────────────────────────────────────────────────────────"
        } | tee -a "$LOG_FILE"

        TRAIN_OK=true
        "$PYTHON" scripts/run_baseline.py \
            --antennas "$N_ANT" \
            --data     "$PKL_FILE" \
            --holdout  "$HOLDOUT" \
            2>&1 | tee "$EXP_DIR/training.log" \
            || TRAIN_OK=false

        if [[ "$TRAIN_OK" == false ]]; then
            MSG="FAILED: $EXP_KEY"
            echo "  *** $MSG ***" | tee -a "$LOG_FILE"
            echo "$MSG"           >> "$ERRORS_LOG"
            FAILED_RUNS+=("$EXP_KEY")
            echo ""
            continue
        fi

        # ── Locate the saved_models directory just created ────────────────────
        SAVED_DIR=$(ls -d saved_models/*_phase_relock_"${N_ANT}"ant 2>/dev/null \
                    | sort -r | head -1 || true)

        if [[ -z "$SAVED_DIR" || ! -d "$SAVED_DIR" ]]; then
            MSG="Could not locate saved model directory for '$EXP_KEY'"
            echo "  WARNING: $MSG" | tee -a "$LOG_FILE"
            echo "WARNING: $MSG"   >> "$ERRORS_LOG"
            FAILED_RUNS+=("$EXP_KEY")
            echo ""
            continue
        fi

        echo "  Saved dir : $SAVED_DIR" | tee -a "$LOG_FILE"

        # ── Copy artifacts ────────────────────────────────────────────────────
        [[ -f "$SAVED_DIR/metrics.json" ]]       && cp "$SAVED_DIR/metrics.json"       "$EXP_DIR/metrics/"
        [[ -f "$SAVED_DIR/distances.npy" ]]      && cp "$SAVED_DIR/distances.npy"      "$EXP_DIR/metrics/"
        [[ -f "$SAVED_DIR/predictions_3d.png" ]] && cp "$SAVED_DIR/predictions_3d.png" "$EXP_DIR/images/"

        # ── CDF plot ──────────────────────────────────────────────────────────
        "$PYTHON" scripts/visualize.py \
            --mode      cdf \
            --distances "$EXP_DIR/metrics/distances.npy" \
            --save      "$EXP_DIR/images/cdf.png" \
            2>&1 >> "$EXP_DIR/training.log" \
            || echo "  WARNING: CDF plot failed for $EXP_KEY" | tee -a "$LOG_FILE"

        # ── Append row to summary CSV ─────────────────────────────────────────
        METRICS_ROW=$("$PYTHON" - "$EXP_DIR/metrics/metrics.json" <<'PYEOF'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        m = json.load(f)
    p = m.get("percentiles_cm", {})
    cols = [
        str(m.get("mean_distance_error_cm", "NA")),
        str(m.get("std_cm", "NA")),
        str(p.get("p25", "NA")),
        str(p.get("p50", "NA")),
        str(p.get("p75", "NA")),
        str(p.get("p90", "NA")),
        str(p.get("p95", "NA")),
        str(p.get("p99", "NA")),
    ]
    print(",".join(cols))
except Exception as e:
    print("NA,NA,NA,NA,NA,NA,NA,NA")
    print(f"[warn] {e}", file=sys.stderr)
PYEOF
        )
        echo "${TRAJ},${N_ANT},${METRICS_ROW}" >> "$SUMMARY_CSV"

        SUCCESSFUL_RUNS+=("$EXP_KEY")
        echo "  Done ✓" | tee -a "$LOG_FILE"
        echo ""
    done
done

# ─── Comparison plot ─────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "  Generating comparison plots ..."                          | tee -a "$LOG_FILE"

"$PYTHON" scripts/compare_runs.py \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG_FILE" \
    || echo "  WARNING: compare_runs.py failed" | tee -a "$LOG_FILE"

# ─── Final summary ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  Phase Relock complete"
echo "  Base name  : $BASE_NAME"
echo "  Successful : ${#SUCCESSFUL_RUNS[@]}"
for r in "${SUCCESSFUL_RUNS[@]}"; do echo "    ✓  $r"; done
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
echo "  Failed     : ${#FAILED_RUNS[@]}"
for r in "${FAILED_RUNS[@]}"; do echo "    ✗  $r"; done
echo "  (see $ERRORS_LOG)"
fi
if [[ ${#SKIPPED_RUNS[@]} -gt 0 ]]; then
echo "  Skipped    : ${#SKIPPED_RUNS[@]}"
for r in "${SKIPPED_RUNS[@]}"; do echo "    -  $r"; done
fi
echo "  Results    : $RESULTS_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if command -v column &>/dev/null; then
    column -t -s',' "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi
