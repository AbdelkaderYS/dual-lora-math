#!/bin/bash
# ============================================================
# run_experiments.sh
#
# Usage:
#   bash scripts/run_experiments.sh
#   bash scripts/run_experiments.sh --dry-run
#   bash scripts/run_experiments.sh --method=dual_lora
# ============================================================

DRY_RUN=false
ONLY_METHOD=""
for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN=true ;;
        --method=*)   ONLY_METHOD="${arg#--method=}" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source /mnt/home/djagbapr/envs/dual_lora/bin/activate

SEEDS=(42 123 456)
METHODS=("dual_lora"                     "lora_baseline")
CONFIGS=("configs/dual_lora_config.yaml" "configs/lora_baseline_config.yaml")

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
mkdir -p logs experiments/results

echo "============================================"
echo "Experiments — $(date)"
echo "Seeds   : ${SEEDS[*]}"
echo "GPUs    : $NUM_GPUS"
echo "Dry run : $DRY_RUN"
echo "============================================"

FAILED=()

for i in "${!METHODS[@]}"; do
    METHOD=${METHODS[$i]}
    CONFIG=${CONFIGS[$i]}

    if [ -n "$ONLY_METHOD" ] && [ "$METHOD" != "$ONLY_METHOD" ]; then
        continue
    fi

    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${METHOD}_seed${SEED}"
        OUT_DIR="experiments/results/${RUN_NAME}"
        LOG_FILE="logs/${RUN_NAME}.out"

        echo ""
        echo "─── $RUN_NAME ───────────────────────────────"
        echo "  Config : $CONFIG"
        echo "  Seed   : $SEED"
        echo "  Output : $OUT_DIR"

        if [ -f "$OUT_DIR/eval_results.json" ]; then
            echo "  [SKIP] already done"
            continue
        fi

        [ "$DRY_RUN" = "true" ] && echo "  [DRY RUN]" && continue

        mkdir -p "$OUT_DIR"
        echo "  Start  : $(date)"

        if [ "$NUM_GPUS" -le 1 ]; then
            python src/training/trainer.py \
                --config "$CONFIG" \
                --output_dir "$OUT_DIR" \
                --seed "$SEED" \
                2>&1 | tee "$LOG_FILE"
        else
            accelerate launch \
                --num_processes "$NUM_GPUS" \
                --mixed_precision "bf16" \
                src/training/trainer.py \
                --config "$CONFIG" \
                --output_dir "$OUT_DIR" \
                --seed "$SEED" \
                2>&1 | tee "$LOG_FILE"
        fi

        EXIT_CODE=${PIPESTATUS[0]}
        echo "  End    : $(date) — exit $EXIT_CODE"

        if [ $EXIT_CODE -ne 0 ]; then
            echo "  [ERROR] $RUN_NAME failed"
            FAILED+=("$RUN_NAME")
            continue
        fi

        echo ""
        echo "  → Evaluating $RUN_NAME..."
        python src/evaluation/evaluator.py \
            --checkpoint_path "$OUT_DIR" \
            --dataset both \
            --batch_size 32 \
            2>&1 | tee -a "$LOG_FILE"

        echo "  ✓ $RUN_NAME done"
    done
done

echo ""
echo "============================================"
echo "Done — $(date)"
echo "============================================"
echo ""
echo "Results:"
for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${METHOD}_seed${SEED}"
        RES="experiments/results/${RUN_NAME}/eval_results.json"
        [ -f "$RES" ] && echo "  ✓ $RUN_NAME" || echo "  ✗ $RUN_NAME"
    done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed:"
    for r in "${FAILED[@]}"; do echo "  ✗ $r"; done
fi

[ -f "scripts/paper_results.py" ] && python scripts/paper_results.py