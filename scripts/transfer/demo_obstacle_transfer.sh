#!/bin/bash
# =============================================================================
# Demo Transfer Pipeline: obstacle <-> many_obstacle
# =============================================================================
#
# Runs the complete cross-task transfer experiment between "obstacle" (50
# obstacles) and "many_obstacle" (150 obstacles).
#
# Pipeline steps:
#   Step 0: Verify prerequisites (trained checkpoints exist)
#   Step 1: Zero-shot evaluation (obstacle policy on many_obstacle, and reverse)
#   Step 2: Fine-tuning at multiple budgets
#   Step 3: Aggregate and display results
#
# Environment notes:
#   - Uses LOKI.TRAIN=True so task.py loads XMLs from {WALKER_DIR}/xml_step/0/
#   - Uses modern mujoco bindings (unimal.py), NOT unimal_original.py
#   - Walker list inferred from {WALKER_DIR}/xml/
#   - NEVER set LOKI.FINETUNE=True (requires deprecated mujoco_py)
#
# Usage:
#   bash scripts/transfer/demo_obstacle_transfer.sh [cluster_list] [seed] [num_clusters]
#
# Examples:
#   # Default: clusters 0,2,18 with 20-cluster setup
#   bash scripts/transfer/demo_obstacle_transfer.sh
#
#   # 40-cluster setup
#   bash scripts/transfer/demo_obstacle_transfer.sh "0 5 12 18 25 33" 3429 40
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
CLUSTERS="${1:-0 2 18}"
SEED="${2:-3429}"
NUM_CLUSTERS="${3:-20}"
NUM_WALKER=20
DROP_FREQ=2
NUM_DROP=2

# Fine-tuning budgets (env steps)
FINETUNE_BUDGETS="2e6 5e6 1e7 2e7 5e7"

# Zero-shot episodes
ZERO_SHOT_EPISODES=50

# Paths
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
METAMORPH_DIR="${PROJECT_ROOT}/metamorph"
LOKI_OUTPUT="${METAMORPH_DIR}/output/loki"
TRANSFER_OUTPUT="${METAMORPH_DIR}/output/transfer"

# WandB
export WANDB_PROJECT="LOKI-transfer"

echo "============================================================"
echo "  Demo Transfer Pipeline: obstacle <-> many_obstacle"
echo "============================================================"
echo "  Clusters:      ${CLUSTERS}"
echo "  Seed:          ${SEED}"
echo "  Num clusters:  ${NUM_CLUSTERS}"
echo "  FT budgets:    ${FINETUNE_BUDGETS}"
echo "  Project root:  ${PROJECT_ROOT}"
echo "============================================================"
echo ""

# =============================================================================
# Helper: get source checkpoint directory
# =============================================================================
get_ckpt_dir() {
    local TASK_DIR="$1"
    local CLUSTER="$2"
    echo "${LOKI_OUTPUT}/${TASK_DIR}/kmeans_cluster/${NUM_CLUSTERS}/${CLUSTER}/walker${NUM_WALKER}/freq${DROP_FREQ}/drop${NUM_DROP}/seed${SEED}"
}

# =============================================================================
# Helper: prepare walker directory for transfer
#
# Both xml/ (for walker list inference by train_ppo.py) and xml_step/0/
# (for actual XML loading by task.py with LOKI.TRAIN=True) must contain
# the agent XMLs.
# =============================================================================
prepare_walker_dir() {
    local SRC_CKPT_DIR="$1"   # source training output dir
    local DST_WALKER_DIR="$2"  # destination walker dir (absolute path)

    # Find the latest xml_step iteration
    local LATEST_ITER
    LATEST_ITER=$(ls "${SRC_CKPT_DIR}/xml_step/" | grep -E '^[0-9]+$' | sort -n | tail -1)

    mkdir -p "${DST_WALKER_DIR}/xml"
    mkdir -p "${DST_WALKER_DIR}/xml_step/0"

    # Copy numbered agent XMLs to both locations
    for xml_file in "${SRC_CKPT_DIR}/xml_step/${LATEST_ITER}"/[0-9]*.xml; do
        [ -f "$xml_file" ] || continue
        cp "$xml_file" "${DST_WALKER_DIR}/xml/"
        cp "$xml_file" "${DST_WALKER_DIR}/xml_step/0/"
    done

    local COUNT
    COUNT=$(ls "${DST_WALKER_DIR}/xml/"*.xml 2>/dev/null | wc -l)
    echo "    Prepared ${COUNT} agent XMLs from iter ${LATEST_ITER}"
}

# =============================================================================
# Step 0: Verify prerequisites
# =============================================================================
echo ">>> Step 0: Checking available trained checkpoints"
echo ""

AVAIL_OBS=""    # clusters with obstacle training done
AVAIL_MANY=""   # clusters with many_obstacle training done

for C in $CLUSTERS; do
    OBS_DIR=$(get_ckpt_dir "obstacle" "$C")
    MANY_DIR=$(get_ckpt_dir "many_obstacle" "$C")

    if [ -f "${OBS_DIR}/Unimal-v0.pt" ] && [ -d "${OBS_DIR}/xml_step" ]; then
        ITER=$(ls "${OBS_DIR}/xml_step/" | grep -E '^[0-9]+$' | sort -n | tail -1)
        N=$(ls "${OBS_DIR}/xml_step/${ITER}"/[0-9]*.xml 2>/dev/null | wc -l)
        echo "  [OK] obstacle      C${C}: ${N} XMLs (iter ${ITER})"
        AVAIL_OBS="${AVAIL_OBS} ${C}"
    else
        echo "  [--] obstacle      C${C}: not found"
    fi

    if [ -f "${MANY_DIR}/Unimal-v0.pt" ] && [ -d "${MANY_DIR}/xml_step" ]; then
        ITER=$(ls "${MANY_DIR}/xml_step/" | grep -E '^[0-9]+$' | sort -n | tail -1)
        N=$(ls "${MANY_DIR}/xml_step/${ITER}"/[0-9]*.xml 2>/dev/null | wc -l)
        echo "  [OK] many_obstacle C${C}: ${N} XMLs (iter ${ITER})"
        AVAIL_MANY="${AVAIL_MANY} ${C}"
    else
        echo "  [--] many_obstacle C${C}: not found"
    fi
done

AVAIL_OBS=$(echo "$AVAIL_OBS" | xargs)
AVAIL_MANY=$(echo "$AVAIL_MANY" | xargs)

echo ""
echo "  obstacle -> many_obstacle: [${AVAIL_OBS:-none}]"
echo "  many_obstacle -> obstacle: [${AVAIL_MANY:-none}]"

if [ -z "$AVAIL_OBS" ] && [ -z "$AVAIL_MANY" ]; then
    echo "ERROR: No completed runs found. Exiting."
    exit 1
fi
echo ""

# =============================================================================
# Step 1: Zero-shot evaluation
# =============================================================================
echo ">>> Step 1: Zero-shot cross-task evaluation"
echo ""

run_zero_shot() {
    local SRC_TASK="$1" TGT_TASK="$2" CLUSTER="$3" SRC_CKPT="$4"

    local OUT="${TRANSFER_OUTPUT}/zero_shot/${SRC_TASK}_to_${TGT_TASK}/c${CLUSTER}"
    local WALKER="${OUT}/walkers"

    # Skip if done
    if [ -f "${OUT}/eval_results.json" ]; then
        echo "    [SKIP] ${OUT}/eval_results.json exists"
        return 0
    fi

    # Prepare walkers
    prepare_walker_dir "$SRC_CKPT" "$WALKER"

    # Target task config: both obstacle and many_obstacle use obstacle.yaml
    local EXTRA=""
    if [ "$TGT_TASK" = "many_obstacle" ]; then
        EXTRA="OBJECT.NUM_OBSTACLES 150 ENV_TYPE many_obstacle"
    fi

    echo "    Evaluating ${ZERO_SHOT_EPISODES} episodes..."
    cd "${METAMORPH_DIR}"
    MUJOCO_GL=egl PYTHONPATH=./ python tools/eval_zero_shot.py \
        --cfg ./configs/obstacle.yaml \
        --checkpoint "${SRC_CKPT}/Unimal-v0.pt" \
        --walker_dir "${WALKER}" \
        --num_episodes ${ZERO_SHOT_EPISODES} \
        --out_dir "${OUT}" \
        $EXTRA \
        2>&1 | tail -8
    cd "${PROJECT_ROOT}"

    [ -f "${OUT}/eval_results.json" ] && echo "    [DONE]" || echo "    [FAIL]"
}

# obstacle -> many_obstacle
if [ -n "$AVAIL_OBS" ]; then
    echo "--- obstacle -> many_obstacle ---"
    for C in $AVAIL_OBS; do
        echo "  C${C}:"
        run_zero_shot "obstacle" "many_obstacle" "$C" "$(get_ckpt_dir obstacle "$C")"
    done
    echo ""
fi

# many_obstacle -> obstacle
if [ -n "$AVAIL_MANY" ]; then
    echo "--- many_obstacle -> obstacle ---"
    for C in $AVAIL_MANY; do
        echo "  C${C}:"
        run_zero_shot "many_obstacle" "obstacle" "$C" "$(get_ckpt_dir many_obstacle "$C")"
    done
    echo ""
fi

# =============================================================================
# Step 2: Fine-tuning at multiple budgets
# =============================================================================
echo ">>> Step 2: Fine-tuning budget sweep"
echo ""

run_finetune() {
    local SRC_TASK="$1" TGT_TASK="$2" CLUSTER="$3" SRC_CKPT="$4" BUDGET="$5"

    local BASE="${TRANSFER_OUTPUT}/finetune/${SRC_TASK}_to_${TGT_TASK}/c${CLUSTER}"
    local OUT="${BASE}/steps_${BUDGET}/seed${SEED}"
    local WALKER="${BASE}/walkers"

    # Skip if done
    if [ -f "${METAMORPH_DIR}/${OUT}/Unimal-v0_results.json" ]; then
        echo "    [SKIP] ${BUDGET} already done"
        return 0
    fi

    # Prepare walkers (only once per cluster, shared across budgets)
    if [ ! -d "${METAMORPH_DIR}/${WALKER}/xml_step/0" ]; then
        prepare_walker_dir "$SRC_CKPT" "${METAMORPH_DIR}/${WALKER}"
    fi

    # Target config
    local EXTRA=""
    if [ "$TGT_TASK" = "many_obstacle" ]; then
        EXTRA="OBJECT.NUM_OBSTACLES 150"
    fi

    # Deterministic WandB run ID
    export WANDB_RUN_ID="transfer-${SRC_TASK}-to-${TGT_TASK}-c${CLUSTER}-steps${BUDGET}-s${SEED}"

    echo "    ${BUDGET} steps (wandb: ${WANDB_RUN_ID})"
    cd "${METAMORPH_DIR}"
    MUJOCO_GL=egl PYTHONPATH=./ python tools/train_ppo.py \
        --cfg ./configs/obstacle.yaml \
        LOKI.TRAIN True \
        OUT_DIR "$OUT" \
        ENV.WALKER_DIR "$WALKER" \
        PPO.CHECKPOINT_PATH "${SRC_CKPT}/Unimal-v0.pt" \
        MODEL.FINETUNE.FULL_MODEL True \
        ENV_TYPE "$TGT_TASK" \
        PPO.MAX_STATE_ACTION_PAIRS "$BUDGET" \
        RNG_SEED "$SEED" \
        $EXTRA \
        2>&1 | tail -3
    cd "${PROJECT_ROOT}"

    if [ -f "${METAMORPH_DIR}/${OUT}/Unimal-v0_results.json" ]; then
        echo "    [DONE] ${BUDGET}"
    else
        echo "    [FAIL] ${BUDGET}"
    fi
}

# obstacle -> many_obstacle
if [ -n "$AVAIL_OBS" ]; then
    echo "--- obstacle -> many_obstacle (fine-tune) ---"
    for C in $AVAIL_OBS; do
        echo "  C${C}:"
        for B in $FINETUNE_BUDGETS; do
            run_finetune "obstacle" "many_obstacle" "$C" "$(get_ckpt_dir obstacle "$C")" "$B"
        done
    done
    echo ""
fi

# many_obstacle -> obstacle
if [ -n "$AVAIL_MANY" ]; then
    echo "--- many_obstacle -> obstacle (fine-tune) ---"
    for C in $AVAIL_MANY; do
        echo "  C${C}:"
        for B in $FINETUNE_BUDGETS; do
            run_finetune "many_obstacle" "obstacle" "$C" "$(get_ckpt_dir many_obstacle "$C")" "$B"
        done
    done
    echo ""
fi

# =============================================================================
# Step 3: Aggregate results
# =============================================================================
echo ">>> Step 3: Collecting results"
echo ""

python3 "${PROJECT_ROOT}/scripts/transfer/aggregate_demo_results.py" \
    --transfer_dir "${TRANSFER_OUTPUT}" \
    --loki_dir "${LOKI_OUTPUT}" \
    --clusters "${AVAIL_OBS}" \
    --clusters_reverse "${AVAIL_MANY}" \
    --num_clusters "${NUM_CLUSTERS}" \
    --seed "${SEED}"

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"
