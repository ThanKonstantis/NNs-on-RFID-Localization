#!/usr/bin/env bash
# run_combine_2d_3d.sh
# Train 2-D models (or reuse an existing results_2D run) and combine
# per-antenna predictions into 3-D position estimates via circle intersection.
#
# Usage:
#   ./run_combine_2d_3d.sh <run_name> [OPTIONS]
#
# Options:
#   --data          PATH   Experiment_Data.pkl (default: Experiments/Experiment_Data.pkl)
#   --results-2d    PATH   Reuse an existing results_2D/<run_name>/ (skips training)
#   --n-antennas    LIST   Comma-separated antenna counts to combine (default: 2,3,4)
#   --device        STR    pytorch device: cpu / cuda (default: auto-detect)
#   --skip-train           Skip 2D training; requires results_2D/<run_name>/ to exist
#
# Output structure:
#   results_3D_combined/<run_name>/
#   ├── run.log
#   ├── summary.csv
#   ├── comparison_cdf.png
#   ├── summary_bar.png
#   └── <N>ant/
#       ├── metrics/
#       │   ├── metrics.json
#       │   └── distances.npy
#       └── images/
#           ├── cdf.png
#           └── predictions_3d.png

set -euo pipefail

# ─── Locate project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Defaults ───────────────────────────────────────────────────────────────
DATA="Experiments/Experiment_Data.pkl"
RESULTS_2D_DIR=""
N_ANTENNAS_LIST=(2 3 4)
DEVICE_ARG=""
SKIP_TRAIN=false

# ─── Usage ──────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \?//' | head -30
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

RUN_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)         DATA="$2";                                           shift 2 ;;
        --results-2d)   RESULTS_2D_DIR="$2";                                shift 2 ;;
        --n-antennas)   IFS=',' read -ra N_ANTENNAS_LIST <<< "$2";          shift 2 ;;
        --device)       DEVICE_ARG="--device $2";                           shift 2 ;;
        --skip-train)   SKIP_TRAIN=true;                                     shift   ;;
        -h|--help)      usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ─── Validate environment ────────────────────────────────────────────────────
if [[ ! -f "$DATA" ]]; then
    echo "ERROR: Data file '$DATA' not found." >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python interpreter not found." >&2
    echo "       Set PYTHON env var to the correct path." >&2
    exit 1
fi

# ─── Determine / create results_2D directory ─────────────────────────────────
if [[ -n "$RESULTS_2D_DIR" ]]; then
    # User supplied an explicit path
    if [[ ! -d "$RESULTS_2D_DIR" ]]; then
        echo "ERROR: --results-2d '$RESULTS_2D_DIR' not found." >&2
        exit 1
    fi
elif [[ "$SKIP_TRAIN" == true ]]; then
    RESULTS_2D_DIR="results_2D/${RUN_NAME}"
    if [[ ! -d "$RESULTS_2D_DIR" ]]; then
        echo "ERROR: --skip-train set but '$RESULTS_2D_DIR' does not exist." >&2
        exit 1
    fi
else
    # Run the 2-D experiment suite first
    RESULTS_2D_DIR="results_2D/${RUN_NAME}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "  Phase 1: Training 2-D models"
    echo "  Run name : $RUN_NAME"
    echo "  Output   : $RESULTS_2D_DIR"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    ./run_experiments_2D.sh "$RUN_NAME" --data "$DATA" ${DEVICE_ARG}
    echo ""
fi

echo "Using 2-D results from: $RESULTS_2D_DIR"

# ─── Setup 3D combined output ────────────────────────────────────────────────
OUT_DIR="results_3D_combined/${RUN_NAME}"
mkdir -p "$OUT_DIR"
export MPLBACKEND=Agg

LOG_FILE="$OUT_DIR/run.log"
SUMMARY_CSV="$OUT_DIR/summary.csv"

echo "model,mean_error_cm,std_cm,p25_cm,p50_cm,p75_cm,p90_cm,p95_cm,p99_cm" \
    > "$SUMMARY_CSV"

# ─── Banner ──────────────────────────────────────────────────────────────────
banner() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "  Phase 2: 2-D → 3-D combination"
    echo "  Run name   : $RUN_NAME"
    echo "  Date       : $(date)"
    echo "  Source 2D  : $RESULTS_2D_DIR"
    echo "  N-antennas : ${N_ANTENNAS_LIST[*]}"
    echo "  Output     : $OUT_DIR"
    echo "╚══════════════════════════════════════════════════════════╝"
}

banner | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ─── Per-antenna-count combination loop ──────────────────────────────────────
TOTAL=${#N_ANTENNAS_LIST[@]}
IDX=0
SUCCESSFUL=()
FAILED=()

for N_ANT in "${N_ANTENNAS_LIST[@]}"; do
    IDX=$((IDX + 1))
    LABEL="${N_ANT}ant"
    SEP="──────────────────────────────────────────────────────────"

    echo "$SEP"                                          | tee -a "$LOG_FILE"
    echo "  [$IDX/$TOTAL]  n-antennas = $N_ANT"         | tee -a "$LOG_FILE"
    echo "$SEP"                                          | tee -a "$LOG_FILE"

    ANT_OUT="$OUT_DIR/$LABEL"
    mkdir -p "$ANT_OUT/metrics" "$ANT_OUT/images"

    COMBINE_OK=true
    "$PYTHON" scripts/combine_2d_to_3d.py \
        --results-dir "$RESULTS_2D_DIR" \
        --data        "$DATA" \
        --n-antennas  "$N_ANT" \
        --output-dir  "$ANT_OUT" \
        ${DEVICE_ARG} \
        2>&1 | tee "$ANT_OUT/combine.log" \
        || COMBINE_OK=false

    if [[ "$COMBINE_OK" == false ]]; then
        MSG="FAILED: combination for n-antennas=$N_ANT"
        echo "  *** $MSG ***" | tee -a "$LOG_FILE"
        FAILED+=("$LABEL")
        continue
    fi

    # ── CDF plot ──────────────────────────────────────────────────────────────
    CDF_OK=true
    "$PYTHON" scripts/visualize.py \
        --mode      cdf \
        --distances "$ANT_OUT/metrics/distances.npy" \
        --save      "$ANT_OUT/images/cdf.png" \
        2>&1 >> "$ANT_OUT/combine.log" \
        || CDF_OK=false

    if [[ "$CDF_OK" == false ]]; then
        echo "  WARNING: CDF plot failed for n-antennas=$N_ANT" | tee -a "$LOG_FILE"
    else
        echo "  CDF saved : $ANT_OUT/images/cdf.png" | tee -a "$LOG_FILE"
    fi

    # ── Append to summary CSV ─────────────────────────────────────────────────
    METRICS_ROW=$("$PYTHON" - "$ANT_OUT/metrics/metrics.json" <<'PYEOF'
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
    echo "${LABEL},${METRICS_ROW}" >> "$SUMMARY_CSV"

    SUCCESSFUL+=("$LABEL")
    echo "  Done ✓" | tee -a "$LOG_FILE"
    echo ""
done

# ─── Comparison plots across n-antenna variants ───────────────────────────────
echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "  Generating comparison plots ..."                          | tee -a "$LOG_FILE"

"$PYTHON" scripts/compare_runs.py \
    --results-dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE" \
    || echo "  WARNING: compare_runs.py failed" | tee -a "$LOG_FILE"

# ─── Final summary ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  Run complete: $RUN_NAME"
echo "  Successful : ${#SUCCESSFUL[@]}  →  ${SUCCESSFUL[*]:-none}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
echo "  Failed     : ${#FAILED[@]}  →  ${FAILED[*]}"
fi
echo "  Results    : $OUT_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if command -v column &>/dev/null; then
    column -t -s',' "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi
