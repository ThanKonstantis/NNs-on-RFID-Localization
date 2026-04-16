#!/usr/bin/env bash
# run_experiments_2D.sh
# Train and evaluate every 2-D architecture, replicating the experiment
# sequence from Neural_Scripts_RAW_2D_WITH_CV.ipynb.
#
# Usage:
#   ./run_experiments_2D.sh <run_name> [OPTIONS]
#
# Options:
#   --data      PATH    Directory with final_tensor.npy / final_labels.npy
#                       (default: Experiments/Raw_Data_Single_Antenna_0)
#   --models    LIST    Comma-separated model keys to run (default: all)
#   --device    STR     pytorch device: cpu / cuda (default: auto-detect)
#
# Experiment schedule (matches notebook):
#   phase_relock                       — 2-D nonlinear baseline
#   simple_adam                        — SimpleModel, Adam,    100 epochs
#   simple_rmsprop                     — SimpleModel, RMSprop, 100 epochs
#   simple_sgd                         — SimpleModel, SGD,     100 epochs
#   relu_128                           — ReLUModel h=128,   Adam, 200 epochs
#   relu_256                           — ReLUModel h=256,   Adam, 200 epochs
#   relu_64                            — ReLUModel h=64,    Adam, 200 epochs
#   leaky_relu                         — LeakyReLUModel h=128, Adam, 200 epochs
#   leaky_relu4                        — LeakyReLUModel4 h=64, Adam, 200 epochs
#   leaky_relu2                        — LeakyReLUModel2 h=64, Adam, 200 epochs
#   leaky_relu2_rms                    — LeakyReLUModel2 h=64, RMSprop, 200 epochs
#   tanh                               — TanhModel h=64,    Adam, 200 epochs
#   sigmoid                            — SigmoidModel h=64, Adam, 200 epochs
#   leaky_relu_drop                    — LeakyReLUModelDropout h=128, Adam lr=1e-2 wd=1e-4, 350 epochs
#   relu_drop                          — ReLUModelDropout h=256,     Adam lr=1e-3, 350 epochs
#   rnn                                — EnhancedRNN,  Adam, 200 epochs
#   best                               — FlexibleMLP (1540,1024,512,256,128), Adam lr=1e-3 bs=64, 200 epochs
#
# Output structure:
#   results_2D/<run_name>/
#   ├── run.log
#   ├── summary.csv
#   ├── errors.log
#   └── <experiment_key>/
#       ├── training.log
#       ├── model.pth
#       ├── metrics/
#       │   ├── metrics.json
#       │   └── distances.npy
#       └── images/
#           ├── cdf.png
#           ├── predictions_2d.pdf
#           └── fold_N_loss.png

set -euo pipefail

# ─── Locate project root ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Defaults ───────────────────────────────────────────────────────────────
DATA="Experiments/Raw_Data_Single_Antenna_0"
SELECTED_MODELS=()
DEVICE_ARG=""

DEFAULT_MODELS=(
    phase_relock
    simple_adam
    simple_rmsprop
    simple_sgd
    relu_128
    relu_256
    relu_64
    leaky_relu
    leaky_relu4
    leaky_relu2
    leaky_relu2_rms
    leaky_relu_drop
    relu_drop
    best
)

# ─── Usage ──────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \?//' | head -40
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

RUN_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)     DATA="$2";                                        shift 2 ;;
        --models)   IFS=',' read -ra SELECTED_MODELS <<< "$2";       shift 2 ;;
        --device)   DEVICE_ARG="--device $2";                        shift 2 ;;
        -h|--help)  usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [[ ${#SELECTED_MODELS[@]} -gt 0 ]]; then
    MODELS=("${SELECTED_MODELS[@]}")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# ─── Validate environment ────────────────────────────────────────────────────
RESULTS_DIR="results_2D/${RUN_NAME}"

if [[ -d "$RESULTS_DIR" ]]; then
    echo "ERROR: '$RESULTS_DIR' already exists." >&2
    echo "       Choose a different run name or delete the existing directory." >&2
    exit 1
fi

if [[ ! -d "$DATA" ]]; then
    echo "ERROR: Data directory '$DATA' not found." >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python interpreter not found." >&2
    echo "       Set PYTHON env var to the correct path." >&2
    exit 1
fi

# ─── Setup output ────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
export MPLBACKEND=Agg

LOG_FILE="$RESULTS_DIR/run.log"
SUMMARY_CSV="$RESULTS_DIR/summary.csv"
ERRORS_LOG="$RESULTS_DIR/errors.log"

echo "model,mean_error_cm,std_cm,p25_cm,p50_cm,p75_cm,p90_cm,p95_cm,p99_cm" \
    > "$SUMMARY_CSV"

# ─── Banner ──────────────────────────────────────────────────────────────────
banner() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "  Run name  : $RUN_NAME"
    echo "  Date      : $(date)"
    echo "  Data      : $DATA"
    echo "  Models    : ${MODELS[*]}"
    echo "  Results   : $RESULTS_DIR"
    echo "╚══════════════════════════════════════════════════════════╝"
}

banner | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ─── Shared flags for train_2d.py ────────────────────────────────────────────
BASE="--data $DATA $DEVICE_ARG"

# ─── Per-experiment definitions ──────────────────────────────────────────────
# Each entry: "model_key|train_2d.py args"
# phase_relock is handled separately via run_baseline_2d.py

declare -A EXPERIMENT_ARGS
EXPERIMENT_ARGS["phase_relock"]="BASELINE"
EXPERIMENT_ARGS["simple_adam"]="--model simple   --epochs 100  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["simple_rmsprop"]="--model simple   --epochs 100  --folds 5 --optimizer rmsprop --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["simple_sgd"]="--model simple   --epochs 100  --folds 5 --optimizer sgd     --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["relu_128"]="--model relu     --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["relu_256"]="--model relu     --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32 --hidden 256"
EXPERIMENT_ARGS["relu_64"]="--model relu     --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32 --hidden 64"
EXPERIMENT_ARGS["leaky_relu"]="--model leaky_relu  --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["leaky_relu4"]="--model leaky_relu4 --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["leaky_relu2"]="--model leaky_relu2 --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["leaky_relu2_rms"]="--model leaky_relu2 --epochs 200  --folds 5 --optimizer rmsprop --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["tanh"]="--model tanh     --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["sigmoid"]="--model sigmoid  --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["leaky_relu_drop"]="--model leaky_relu_drop --epochs 350 --folds 5 --optimizer adam --lr 1e-2 --weight-decay 1e-4 --batch-size 32 --hidden 128"
EXPERIMENT_ARGS["relu_drop"]="--model relu_drop       --epochs 350 --folds 5 --optimizer adam --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["rnn"]="--model rnn      --epochs 200  --folds 5 --optimizer adam    --lr 1e-3 --batch-size 32"
EXPERIMENT_ARGS["best"]="--model mlp --hidden 1540,1024,512,256,128 --epochs 200 --folds 5 --optimizer adam --lr 1e-3 --batch-size 64"

# ─── Per-model training loop ─────────────────────────────────────────────────
FAILED_MODELS=()
SUCCESSFUL_MODELS=()
TOTAL=${#MODELS[@]}
IDX=0

for EXP_KEY in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    SEP="──────────────────────────────────────────────────────────"

    echo "$SEP"                                    | tee -a "$LOG_FILE"
    echo "  [$IDX/$TOTAL]  Experiment: $EXP_KEY"  | tee -a "$LOG_FILE"
    echo "$SEP"                                    | tee -a "$LOG_FILE"

    if [[ -z "${EXPERIMENT_ARGS[$EXP_KEY]+x}" ]]; then
        MSG="Unknown experiment key: $EXP_KEY"
        echo "  WARNING: $MSG" | tee -a "$LOG_FILE"
        echo "WARNING: $MSG"   >> "$ERRORS_LOG"
        FAILED_MODELS+=("$EXP_KEY")
        continue
    fi

    EXP_DIR="$RESULTS_DIR/$EXP_KEY"
    mkdir -p "$EXP_DIR/metrics" "$EXP_DIR/images"

    TRAIN_OK=true
    XARGS="${EXPERIMENT_ARGS[$EXP_KEY]}"

    if [[ "$XARGS" == "BASELINE" ]]; then
        # ── Phase Relock baseline ─────────────────────────────────────────────
        "$PYTHON" scripts/run_baseline_2d.py \
            --data "$DATA" \
            2>&1 | tee "$EXP_DIR/training.log" \
            || TRAIN_OK=false
    else
        # ── Neural network experiment ─────────────────────────────────────────
        # shellcheck disable=SC2086
        "$PYTHON" scripts/train_2d.py \
            $XARGS \
            $BASE \
            --quiet \
            2>&1 | tee "$EXP_DIR/training.log" \
            || TRAIN_OK=false
    fi

    if [[ "$TRAIN_OK" == false ]]; then
        MSG="FAILED: $EXP_KEY"
        echo "  *** $MSG ***"   | tee -a "$LOG_FILE"
        echo "$MSG"             >> "$ERRORS_LOG"
        FAILED_MODELS+=("$EXP_KEY")
        continue
    fi

    # ── Locate the saved_models directory just created ────────────────────────
    # Phase relock dirs end in _phase_relock_2D; NN dirs end in _2D
    if [[ "$XARGS" == "BASELINE" ]]; then
        SAVED_DIR=$(ls -d saved_models/*_phase_relock_2D 2>/dev/null \
                    | sort -r | head -1 || true)
    else
        # Extract model key from --model arg and build pattern
        MODEL_KEY=$(echo "$XARGS" | grep -oP '(?<=--model )\S+')
        SAVED_DIR=$(ls -d saved_models/*_"${MODEL_KEY}"*_2D 2>/dev/null \
                    | sort -r | head -1 || true)
    fi

    if [[ -z "$SAVED_DIR" || ! -d "$SAVED_DIR" ]]; then
        MSG="Could not locate saved model directory for '$EXP_KEY'"
        echo "  WARNING: $MSG"  | tee -a "$LOG_FILE"
        echo "WARNING: $MSG"    >> "$ERRORS_LOG"
        FAILED_MODELS+=("$EXP_KEY")
        continue
    fi

    echo "  Saved dir : $SAVED_DIR" | tee -a "$LOG_FILE"

    # ── Copy artifacts ────────────────────────────────────────────────────────
    [[ -f "$SAVED_DIR/metrics.json" ]]       && cp "$SAVED_DIR/metrics.json"       "$EXP_DIR/metrics/"
    [[ -f "$SAVED_DIR/distances.npy" ]]      && cp "$SAVED_DIR/distances.npy"      "$EXP_DIR/metrics/"
    [[ -f "$SAVED_DIR/model.pth" ]]          && cp "$SAVED_DIR/model.pth"          "$EXP_DIR/"
    [[ -f "$SAVED_DIR/predictions_2d.pdf" ]] && cp "$SAVED_DIR/predictions_2d.pdf" "$EXP_DIR/images/"
    [[ -f "$SAVED_DIR/predictions_2d.png" ]] && cp "$SAVED_DIR/predictions_2d.png" "$EXP_DIR/images/"
    for f in "$SAVED_DIR"/fold_*_loss.png; do
        [[ -f "$f" ]] && cp "$f" "$EXP_DIR/images/"
    done

    # ── CDF plot ──────────────────────────────────────────────────────────────
    CDF_OK=true
    "$PYTHON" scripts/visualize.py \
        --mode      cdf \
        --distances "$EXP_DIR/metrics/distances.npy" \
        --save      "$EXP_DIR/images/cdf.png" \
        2>&1 >> "$EXP_DIR/training.log" \
        || CDF_OK=false

    if [[ "$CDF_OK" == false ]]; then
        echo "  WARNING: CDF plot failed for $EXP_KEY" | tee -a "$LOG_FILE"
    fi

    # ── Append to summary CSV ─────────────────────────────────────────────────
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
    echo "  Done ✓" | tee -a "$LOG_FILE"
    echo ""
done

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
