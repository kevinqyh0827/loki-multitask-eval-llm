#!/bin/bash
# Master orchestrator for transfer learning experiments.
#
# Launches all conditions across budgets and seeds with GPU scheduling.
# Follows a three-phase approach:
#   Phase 0: Train source policies on push_box_incline and bump (pre-requisite)
#   Phase 1: Zero-shot evaluation (cheap, fast)
#   Phase 2: All fine-tune and scratch conditions with budget sweep
#
# Usage:
#   bash scripts/run_transfer_experiments.sh [max_concurrent] [gpu_id]
#
# Arguments:
#   max_concurrent: Max concurrent jobs (default: 4)
#   gpu_id:         GPU device ID to use (default: 0)

set -e

MAX_CONCURRENT=${1:-4}
GPU_ID=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU_ID

# ==================== Configuration ====================

# Source: best ft morphology (cluster 18, agent 0)
FT_SOURCE_DIR="./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429"
MORPH_XML_DIR="${FT_SOURCE_DIR}/xml_step/1218"

# Target task
TARGET="incline"

# Budget sweep (state-action pairs)
BUDGETS=("1e6" "2e6" "5e6" "1e7" "2e7" "5e7")

# Seeds for statistical significance
SEEDS=(3429 1409 42)

# Source policies for cross-task transfer
PBI_SOURCE_DIR="./output/transfer/source_policies/push_box_incline/seed_3429"
BUMP_SOURCE_DIR="./output/transfer/source_policies/bump/seed_3429"

# ==================== Helper Functions ====================

wait_for_slot() {
    # Wait until number of background jobs is below MAX_CONCURRENT
    while [ $(jobs -rp | wc -l) -ge $MAX_CONCURRENT ]; do
        sleep 10
    done
}

count_active_jobs() {
    jobs -rp | wc -l
}

# ==================== Phase 0: Source Policy Training ====================
echo "================================================================"
echo "PHASE 0: Training source policies for cross-task transfer"
echo "================================================================"

# Train push_box_incline source policy (if not already done)
if [ ! -f "metamorph/${PBI_SOURCE_DIR}/Unimal-v0.pt" ]; then
    echo "Training push_box_incline source policy..."
    bash scripts/run_source_training.sh push_box_incline "$MORPH_XML_DIR" 1e8 3429
else
    echo "push_box_incline source policy already exists."
fi

# Train bump source policy (if not already done)
if [ ! -f "metamorph/${BUMP_SOURCE_DIR}/Unimal-v0.pt" ]; then
    echo "Training bump source policy..."
    bash scripts/run_source_training.sh bump "$MORPH_XML_DIR" 1e8 3429
else
    echo "bump source policy already exists."
fi

echo ""
echo "Phase 0 complete. Source policies ready."

# ==================== Phase 1: Zero-Shot Evaluation ====================
echo ""
echo "================================================================"
echo "PHASE 1: Zero-shot evaluation"
echo "================================================================"

bash scripts/eval_zero_shot.sh "$FT_SOURCE_DIR" $TARGET 50 0

echo ""
echo "Phase 1 complete."

# ==================== Phase 2: Budget Sweep ====================
echo ""
echo "================================================================"
echo "PHASE 2: Fine-tune and scratch conditions with budget sweep"
echo "================================================================"
echo "Conditions: 5 (finetune_ft, finetune_pbi, finetune_bump, scratch_transformer, scratch_mlp)"
echo "Budgets: ${BUDGETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max concurrent: $MAX_CONCURRENT"
echo ""

TOTAL_JOBS=0
COMPLETED_JOBS=0
FAILED_JOBS=0

for BUDGET in "${BUDGETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "--- Budget: $BUDGET, Seed: $SEED ---"

        # Condition 2: Fine-tune from ft (LOKI DLS, sim=0.45 to incline)
        wait_for_slot
        bash scripts/run_finetune_single.sh finetune $TARGET $BUDGET $SEED \
            "$MORPH_XML_DIR" \
            "${FT_SOURCE_DIR}/Unimal-v0.pt" \
            ActorCritic ft &
        TOTAL_JOBS=$((TOTAL_JOBS + 1))

        # Condition 3: Scratch with Transformer
        wait_for_slot
        bash scripts/run_finetune_single.sh scratch $TARGET $BUDGET $SEED \
            "$MORPH_XML_DIR" &
        TOTAL_JOBS=$((TOTAL_JOBS + 1))

        # Condition 4: Scratch with MLP
        wait_for_slot
        bash scripts/run_finetune_single.sh scratch_mlp $TARGET $BUDGET $SEED \
            "$MORPH_XML_DIR" &
        TOTAL_JOBS=$((TOTAL_JOBS + 1))

        # Condition 5: Fine-tune from push_box_incline (scratch-trained, sim=0.52)
        wait_for_slot
        bash scripts/run_finetune_single.sh finetune $TARGET $BUDGET $SEED \
            "$MORPH_XML_DIR" \
            "metamorph/${PBI_SOURCE_DIR}/Unimal-v0.pt" \
            ActorCritic push_box_incline &
        TOTAL_JOBS=$((TOTAL_JOBS + 1))

        # Condition 6: Fine-tune from bump (scratch-trained, sim=0.42, cross obs group)
        wait_for_slot
        bash scripts/run_finetune_single.sh finetune $TARGET $BUDGET $SEED \
            "$MORPH_XML_DIR" \
            "metamorph/${BUMP_SOURCE_DIR}/Unimal-v0.pt" \
            ActorCritic bump &
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo ""
echo "All $TOTAL_JOBS jobs launched. Waiting for completion..."
wait

echo ""
echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================"
echo "Total jobs: $TOTAL_JOBS"
echo ""
echo "Run analysis:"
echo "  python tools/analyze_transfer_results.py --base_dir metamorph/output/transfer"
echo "  python tools/plot_transfer_curves.py --base_dir metamorph/output/transfer"
