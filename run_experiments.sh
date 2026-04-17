#!/usr/bin/env bash
# run_experiments.sh
# Train and evaluate every architecture and save all results under a
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
#   --models   LIST    Comma-separated experiment keys to run (default: all)
#   --device   STR     pytorch device: cpu / cuda (default: auto-detect)
#
# Experiment schedule:
#   mlp_128_128              — MLP(128, 128)
#   cnn_32_64_128_128        — CNN[32,130],[64,64],[128,32],[128,16],(128,64)
#   mlp_512_256_128_64       — MLP(512, 256, 128, 64)
#   mlp_3080_1540_64         — MLP(3080, 1540, 64)
#   cnn_32_64_128            — CNN[32,110],[64,84],[128,56],(128,64)
#   rnn_96_2_uni             — RNN[96,2,U],(96,48)
#   cnn_16_32_48             — CNN[16,130],[32,64],[48,32],(48,24)
#   cnn_48_64_96             — CNN[48,140],[64,70],[96,35],(96,32)
#   cnn_32_64_128_256        — CNN[32,120],[64,84],[128,56],[256,28],(256,32)
#   rnn_48_4_uni             — RNN[48,4,U],(48,24)
#   rnn_24_6_uni             — RNN[24,6,U],(24,24)
#   phase_relock             — Phase Relock (SAR baseline)
#   rnn_16_2_bi              — RNN[16,2,Bi],(32,16)
#
# Output structure:
#   results/<run_name>/
#   ├── run.log
#   ├── summary.csv
#   ├── errors.log
#   ├── comparison_cdf.png
#   ├── summary_bar.png
#   └── <experiment_key>/
#       ├── training.log
#       ├── model.pth
#       ├── metrics/
#       │   ├── metrics.json
#       │   └── distances.npy
#       └── images/
#           ├── cdf.png
#           ├── predictions_3d.png
#           └── fold_N_loss.png

set -euo pipefail

# ─── Locate project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Defaults ───────────────────────────────────────────────────────────────
ANTENNAS=3
EPOCHS=200
FOLDS=5
LR=0.01
BATCH_SIZE=32
PATIENCE=90
DATA="Experiments/Experiment_Data.pkl"
DEVICE_ARG=""
SELECTED_MODELS=()

DEFAULT_MODELS=(
    mlp_128_128
    cnn_32_64_128_128
    mlp_512_256_128_64
    mlp_3080_1540_64
    cnn_32_64_128
    rnn_96_2_uni
    cnn_16_32_48
    cnn_48_64_96
    cnn_32_64_128_256
    rnn_48_4_uni
    rnn_24_6_uni
    phase_relock
    rnn_16_2_bi
)

# ─── Usage ──────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \?//' | head -50
    exit 1
}

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
        --device)     DEVICE_ARG="--device $2";                          shift 2 ;;
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
export MPLBACKEND=Agg

LOG_FILE="$RESULTS_DIR/run.log"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
ERRORS_LOG="$RESULTS_DIR/errors.log"

echo "model,mean_error_cm,std_cm,p25_cm,p50_cm,p75_cm,p90_cm,p95_cm,p99_cm" \
    > "$SUMMARY_CSV"

# ─── Per-experiment architecture args ────────────────────────────────────────
# Each value is the model-specific portion of the train.py command.
# Training hyperparams (--antennas, --epochs, etc.) are added automatically.
# "BASELINE" entries are routed to run_baseline.py instead of train.py.

declare -A EXPERIMENT_ARGS
EXPERIMENT_ARGS["phase_relock"]="BASELINE"
EXPERIMENT_ARGS["mlp_128_128"]="--model mlp --hidden 128,128"
EXPERIMENT_ARGS["cnn_32_64_128_128"]="--model flexible_cnn --conv-layers 32,130;64,64;128,32;128,16 --hidden 128,64"
EXPERIMENT_ARGS["mlp_512_256_128_64"]="--model mlp --hidden 512,256,128,64"
EXPERIMENT_ARGS["mlp_3080_1540_64"]="--model mlp --hidden 3080,1540,64"
EXPERIMENT_ARGS["cnn_32_64_128"]="--model flexible_cnn --conv-layers 32,110;64,84;128,56 --hidden 128,64"
EXPERIMENT_ARGS["rnn_96_2_uni"]="--model flexible_rnn --rnn-hidden 96 --rnn-layers 2 --hidden 96,48"
EXPERIMENT_ARGS["cnn_16_32_48"]="--model flexible_cnn --conv-layers 16,130;32,64;48,32 --hidden 48,24"
EXPERIMENT_ARGS["cnn_48_64_96"]="--model flexible_cnn --conv-layers 48,140;64,70;96,35 --hidden 96,32"
EXPERIMENT_ARGS["cnn_32_64_128_256"]="--model flexible_cnn --conv-layers 32,120;64,84;128,56;256,28 --hidden 256,32"
EXPERIMENT_ARGS["rnn_48_4_uni"]="--model flexible_rnn --rnn-hidden 48 --rnn-layers 4 --hidden 48,24"
EXPERIMENT_ARGS["rnn_24_6_uni"]="--model flexible_rnn --rnn-hidden 24 --rnn-layers 6 --hidden 24,24"
EXPERIMENT_ARGS["rnn_16_2_bi"]="--model flexible_rnn --rnn-hidden 16 --rnn-layers 2 --bidirectional --hidden 32,16"

# ─── Common training flags ────────────────────────────────────────────────────
COMMON="--antennas $ANTENNAS --epochs $EPOCHS --folds $FOLDS --lr $LR \
        --batch-size $BATCH_SIZE --patience $PATIENCE --data $DATA $DEVICE_ARG --quiet"

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

for EXP_KEY in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    SEPARATOR="──────────────────────────────────────────────────────────"

    echo "$SEPARATOR"                                   | tee -a "$LOG_FILE"
    echo "  [$IDX/$TOTAL]  Experiment: $EXP_KEY"       | tee -a "$LOG_FILE"
    echo "$SEPARATOR"                                   | tee -a "$LOG_FILE"

    if [[ -z "${EXPERIMENT_ARGS[$EXP_KEY]+x}" ]]; then
        MSG="Unknown experiment key: $EXP_KEY"
        echo "  WARNING: $MSG" | tee -a "$LOG_FILE"
        echo "WARNING: $MSG"   >> "$ERRORS_LOG"
        FAILED_MODELS+=("$EXP_KEY")
        continue
    fi

    EXP_DIR="$RESULTS_DIR/$EXP_KEY"
    mkdir -p "$EXP_DIR/metrics" "$EXP_DIR/images"

    XARGS="${EXPERIMENT_ARGS[$EXP_KEY]}"
    TRAIN_OK=true

    # ── Run training / baseline ───────────────────────────────────────────────
    if [[ "$XARGS" == "BASELINE" ]]; then
        if [[ "$ANTENNAS" -lt 2 ]]; then
            echo "  Skipping phase_relock: requires >= 2 antennas" | tee -a "$LOG_FILE"
            echo "${EXP_KEY},NA,NA,NA,NA,NA,NA,NA,NA" >> "$SUMMARY_CSV"
            continue
        fi
        "$PYTHON" scripts/run_baseline.py \
            --antennas "$ANTENNAS" \
            --data     "$DATA" \
            2>&1 | tee "$EXP_DIR/training.log" \
            || TRAIN_OK=false
    else
        # shellcheck disable=SC2086
        "$PYTHON" scripts/train.py \
            $XARGS \
            $COMMON \
            2>&1 | tee "$EXP_DIR/training.log" \
            || TRAIN_OK=false
    fi

    if [[ "$TRAIN_OK" == false ]]; then
        MSG="FAILED: $EXP_KEY"
        echo "  *** $MSG ***"  | tee -a "$LOG_FILE"
        echo "$MSG"            >> "$ERRORS_LOG"
        FAILED_MODELS+=("$EXP_KEY")
        continue
    fi

    # ── Locate the saved_models directory just created ────────────────────────
    if [[ "$XARGS" == "BASELINE" ]]; then
        SAVED_DIR=$(ls -d saved_models/*_phase_relock_"${ANTENNAS}"ant 2>/dev/null \
                    | sort -r | head -1 || true)
    else
        MODEL_KEY=$(echo "$XARGS" | grep -oP '(?<=--model )\S+')
        SAVED_DIR=$(ls -d saved_models/*_"${MODEL_KEY}"_"${ANTENNAS}"ant 2>/dev/null \
                    | sort -r | head -1 || true)
    fi

    if [[ -z "$SAVED_DIR" || ! -d "$SAVED_DIR" ]]; then
        MSG="Could not locate saved model directory for '$EXP_KEY'"
        echo "  WARNING: $MSG" | tee -a "$LOG_FILE"
        echo "WARNING: $MSG"   >> "$ERRORS_LOG"
        FAILED_MODELS+=("$EXP_KEY")
        continue
    fi

    echo "  Saved dir : $SAVED_DIR" | tee -a "$LOG_FILE"

    # ── Copy artifacts ────────────────────────────────────────────────────────
    [[ -f "$SAVED_DIR/metrics.json" ]]       && cp "$SAVED_DIR/metrics.json"       "$EXP_DIR/metrics/"
    [[ -f "$SAVED_DIR/distances.npy" ]]      && cp "$SAVED_DIR/distances.npy"      "$EXP_DIR/metrics/"
    [[ -f "$SAVED_DIR/model.pth" ]]          && cp "$SAVED_DIR/model.pth"          "$EXP_DIR/"
    [[ -f "$SAVED_DIR/predictions_3d.png" ]] && cp "$SAVED_DIR/predictions_3d.png" "$EXP_DIR/images/"
    for f in "$SAVED_DIR"/fold_*_loss.png; do
        [[ -f "$f" ]] && cp "$f" "$EXP_DIR/images/"
    done

    # ── Generate individual CDF plot ──────────────────────────────────────────
    CDF_OK=true
    "$PYTHON" scripts/visualize.py \
        --mode      cdf \
        --distances "$EXP_DIR/metrics/distances.npy" \
        --save      "$EXP_DIR/images/cdf.png" \
        2>&1 >> "$EXP_DIR/training.log" \
        || CDF_OK=false

    if [[ "$CDF_OK" == false ]]; then
        echo "  WARNING: CDF plot failed for $EXP_KEY" | tee -a "$LOG_FILE"
        echo "WARNING: CDF plot failed for $EXP_KEY"   >> "$ERRORS_LOG"
    else
        echo "  CDF saved : $EXP_DIR/images/cdf.png" | tee -a "$LOG_FILE"
    fi

    # ── Append row to summary CSV ─────────────────────────────────────────────
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
    echo "${EXP_KEY},${METRICS_ROW}" >> "$SUMMARY_CSV"

    SUCCESSFUL_MODELS+=("$EXP_KEY")
    echo "  Done ✓"                  | tee -a "$LOG_FILE"
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

if command -v column &>/dev/null; then
    column -t -s',' "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi
