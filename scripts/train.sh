#!/bin/bash
# ============================================================
# train.sh — Generic training launcher
#
# Usage:
#   bash scripts/train.sh configs/dual_lora_config.yaml
#   bash scripts/train.sh configs/dual_lora_config.yaml --debug
#   sbatch scripts/train.sh configs/dual_lora_config.yaml
# ============================================================

set -e

CONFIG="${1:?Usage: $0 <config.yaml> [--debug] [--seed N] [--output_dir DIR]}"
shift

DEBUG=false
SEED=""
OUT_DIR=""
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --debug) DEBUG=true ;;
        --seed=*) SEED="${arg#--seed=}" ;;
        --output_dir=*) OUT_DIR="${arg#--output_dir=}" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate environment
ENV_PATH="/mnt/home/djagbapr/envs/dual_lora"
if [ -f "$ENV_PATH/bin/activate" ]; then
    source "$ENV_PATH/bin/activate"
fi
# Force the venv python (activate may not override system PATH on this cluster)
PYTHON_BIN="$ENV_PATH/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=$(command -v python3)
fi

mkdir -p logs experiments/results

RUN_NAME=$("$PYTHON_BIN" -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('training', {}).get('run_name', cfg.get('experiment', {}).get('name', 'run')))
" 2>/dev/null || echo "run")

LOG_FILE="logs/${RUN_NAME}.out"
echo "Config  : $CONFIG"
echo "Run     : $RUN_NAME"
echo "Debug   : $DEBUG"
echo "Log     : $LOG_FILE"
echo ""

SCRIPT_ARGS=("src/training/trainer.py" "--config" "$CONFIG")
$DEBUG && SCRIPT_ARGS+=("--debug")
[ -n "$SEED" ] && SCRIPT_ARGS+=("--seed" "$SEED")
[ -n "$OUT_DIR" ] && SCRIPT_ARGS+=("--output_dir" "$OUT_DIR")
SCRIPT_ARGS+=("${EXTRA_ARGS[@]}")

NUM_GPUS=$("$PYTHON_BIN" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision "bf16" \
    "${SCRIPT_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "Exit code: $EXIT_CODE"
echo "Log saved: $LOG_FILE"
exit $EXIT_CODE
