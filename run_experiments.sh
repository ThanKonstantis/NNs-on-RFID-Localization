#!/usr/bin/env bash
# run_experiments.sh
# Train and evaluate every NN architecture and save all results under a
# single named run directory.
#
# Usage:
#   ./run_experiments.sh <run_name> [OPTIONS]
#
# Options:
#   --antennas N       Number of antennas to use (default: 3)
#   --epochs   N       Training epochs per model (default: 200)
#   --folds    N       K-fold cross-validation splits (default: 5)
#   --lr       FLOAT   Learning rate (default: 0.01)
#   --batch-size N     Batch size (default: 32)
#   --patience N       Early stopping patience (default: 90)
#   --data     PATH    Experiment data pickle (default: Experiments/Experiment_Data.pkl)
#   --models   LIST    Comma-separated model names to run (default: all MLP models)
#
# Output structure:
#   results/<run_name>/
#   ├── run.log                 Full execution log
#   ├── summary.csv             One row per model (mean, std, percentiles)
#   ├── errors.log              Failed models / warnings (if any)
#   ├── comparison_cdf.png      All CDFs on one chart
#   ├── summary_bar.png         Bar chart of mean errors ± std
#   └── <model_name>/
#       ├── training.log        Stdout from train.py for this model
#       ├── model.pth           Trained model weights
#       ├── metrics/
#       │   ├── metrics.json    Full metrics + run config
#       │   └── distances.npy   Per-sample Euclidean errors (cm)
#       └── images/
#           ├── cdf.png
#           ├── predictions_3d.png   GT vs predicted 3-D scatter (30 points)
#           └── fold_N_loss.png      Normalised train/val loss per fold         CDF of distance errors
#
# Notes:
#   - Uses MPLBACKEND=Agg so no display is required (runs headless).

set -euo pipefail

# ─── Locate project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_MODELS=(simple relu leaky_relu leaky_relu2 leaky_relu4 leaky_relu_drop relu_drop tanh sigmoid mlp cnn rnn)
ANTENNAS=3
EPOCHS=200
FOLDS=5
LR=0.01
BATCH_SIZE=32
PATIENCE=90
DATA="Experiments/Experiment_Data.pkl"
SELECTED_MODELS=()

# ─── Usage ──────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \?//' | sed -n '2,/^Notes/p'
    exit 1
}

# ─── Parse arguments ────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    usage
fi

RUN_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --antennas)   ANTENNAS="$2";                                      shift 2 ;;
        --epochs)     EPOCHS="$2";                                        shift 2 ;;
        --folds)      FOLDS="$2";                                         shift 2 ;;
        --lr)         LR="$2";                                            shift 2 ;;
        --batch-size) BATCH_SIZE="$2";                                    shift 2 ;;
        --patience)   PATIENCE="$2";                                      shift 2 ;;
        --data)       DATA="$2";                                          shift 2 ;;
        --models)     IFS=',' read -ra SELECTED_MODELS <<< "$2";         shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [[ ${#SELECTED_MODELS[@]} -gt 0 ]]; then
    MODELS=("${SELECTED_MODELS[@]}")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# ─── Validate environment ────────────────────────────────────────────────────
RESULTS_DIR="results/${RUN_NAME}"

if [[ -d "$RESULTS_DIR" ]]; then
    echo "ERROR: '$RESULTS_DIR' already exists." >&2
    echo "       Choose a different run name or delete the existing directory." >&2
    exit 1
fi

if [[ ! -f "$DATA" ]]; then
    echo "ERROR: Data file '$DATA' not found." >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python interpreter not found." >&2
    echo "       Set the PYTHON environment variable to the correct path." >&2
    exit 1
fi

# ─── Setup output directories ────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
export MPLBACKEND=Agg   # headless – plt.show() becomes a no-op

LOG_FILE="$RESULTS_DIR/run.log"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
ERRORS_LOG="$RESULTS_DIR/errors.log"

echo "model,mean_error_cm,std_cm,p25_cm,p50_cm,p75_cm,p90_cm,p95_cm,p99_cm" \
    > "$SUMMARY_CSV"

# ─── Print run banner ────────────────────────────────────────────────────────
banner() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "  Run name  : $RUN_NAME"
    echo "  Date      : $(date)"
    echo "  Antennas  : $ANTENNAS   Epochs: $EPOCHS   Folds: $FOLDS"
    echo "  LR        : $LR         Batch : $BATCH_SIZE   Patience: $PATIENCE"
    echo "  Models    : ${MODELS[*]}"
    echo "  Results   : $RESULTS_DIR"
    echo "  Data      : $DATA"
    echo "╚══════════════════════════════════════════════════════════╝"
}

banner | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ─── Per-model training loop ─────────────────────────────────────────────────
FAILED_MODELS=()
SUCCESSFUL_MODELS=()
TOTAL=${#MODELS[@]}
IDX=0

for MODEL in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    SEPARATOR="──────────────────────────────────────────────────────────"

    echo "$SEPARATOR"                              | tee -a "$LOG_FILE"
    echo "  [$IDX/$TOTAL]  Model: $MODEL"         | tee -a "$LOG_FILE"
    echo "$SEPARATOR"                              | tee -a "$LOG_FILE"

    MODEL_DIR="$RESULTS_DIR/$MODEL"
    mkdir -p "$MODEL_DIR/metrics" "$MODEL_DIR/images"

    # ── Run training ──────────────────────────────────────────────────────────
    TRAIN_OK=true
    "$PYTHON" scripts/train.py \
        --model       "$MODEL" \
        --antennas    "$ANTENNAS" \
        --epochs      "$EPOCHS" \
        --folds       "$FOLDS" \
        --lr          "$LR" \
        --batch-size  "$BATCH_SIZE" \
        --patience    "$PATIENCE" \
        --data        "$DATA" \
        --quiet \
        2>&1 | tee "$MODEL_DIR/training.log" \
        || TRAIN_OK=false

    if [[ "$TRAIN_OK" == false ]]; then
        MSG="FAILED training $MODEL (exit code $?)"
        echo "  *** $MSG ***"                  | tee -a "$LOG_FILE"
        echo "$MSG"                            >> "$ERRORS_LOG"
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    # ── Locate the saved_models directory just created ────────────────────────
    # Directory names follow the pattern: YYYYMMDD_HHMMSS_<model>_<N>ant
    # Sorting in reverse alphabetical order gives the newest first.
    SAVED_DIR=$(ls -d saved_models/*_"${MODEL}"_"${ANTENNAS}"ant 2>/dev/null \
                | sort -r | head -1 || true)

    if [[ -z "$SAVED_DIR" || ! -d "$SAVED_DIR" ]]; then
        MSG="Could not locate saved model directory for '$MODEL'"
        echo "  WARNING: $MSG"                | tee -a "$LOG_FILE"
        echo "WARNING: $MSG"                  >> "$ERRORS_LOG"
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    echo "  Saved dir : $SAVED_DIR"           | tee -a "$LOG_FILE"

    # ── Copy artifacts ────────────────────────────────────────────────────────
    [[ -f "$SAVED_DIR/metrics.json" ]]       && cp "$SAVED_DIR/metrics.json"       "$MODEL_DIR/metrics/"
    [[ -f "$SAVED_DIR/distances.npy" ]]      && cp "$SAVED_DIR/distances.npy"      "$MODEL_DIR/metrics/"
    [[ -f "$SAVED_DIR/model.pth" ]]          && cp "$SAVED_DIR/model.pth"          "$MODEL_DIR/"
    [[ -f "$SAVED_DIR/predictions_3d.png" ]] && cp "$SAVED_DIR/predictions_3d.png" "$MODEL_DIR/images/"
    for f in "$SAVED_DIR"/fold_*_loss.png; do
        [[ -f "$f" ]] && cp "$f" "$MODEL_DIR/images/"
    done

    # ── Generate individual CDF plot ──────────────────────────────────────────
    CDF_OK=true
    "$PYTHON" scripts/visualize.py \
        --mode      cdf \
        --distances "$MODEL_DIR/metrics/distances.npy" \
        --save      "$MODEL_DIR/images/cdf.png" \
        2>&1 >> "$MODEL_DIR/training.log" \
        || CDF_OK=false

    if [[ "$CDF_OK" == false ]]; then
        echo "  WARNING: CDF plot failed for $MODEL" | tee -a "$LOG_FILE"
        echo "WARNING: CDF plot failed for $MODEL"   >> "$ERRORS_LOG"
    else
        echo "  CDF saved : $MODEL_DIR/images/cdf.png" | tee -a "$LOG_FILE"
    fi

    # ── Append row to summary CSV ─────────────────────────────────────────────
    METRICS_ROW=$("$PYTHON" - "$MODEL_DIR/metrics/metrics.json" <<'PYEOF'
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
    echo "${MODEL},${METRICS_ROW}" >> "$SUMMARY_CSV"

    SUCCESSFUL_MODELS+=("$MODEL")
    echo "  Done ✓"                           | tee -a "$LOG_FILE"
    echo ""
done

# ─── Comparison plots (all successful models together) ───────────────────────
echo "──────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
echo "  Generating comparison plots ..."                          | tee -a "$LOG_FILE"

"$PYTHON" scripts/compare_runs.py \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG_FILE" \
    || echo "  WARNING: compare_runs.py failed" | tee -a "$LOG_FILE"

# ─── Final summary ───────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "  Run complete: $RUN_NAME"
echo "  Successful : ${#SUCCESSFUL_MODELS[@]}  →  ${SUCCESSFUL_MODELS[*]:-none}"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
echo "  Failed     : ${#FAILED_MODELS[@]}  →  ${FAILED_MODELS[*]}"
echo "  (see $ERRORS_LOG)"
fi
echo "  Results    : $RESULTS_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Print summary table ─────────────────────────────────────────────────────
if command -v column &>/dev/null; then
    column -t -s',' "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi
