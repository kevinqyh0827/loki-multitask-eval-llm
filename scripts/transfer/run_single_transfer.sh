#!/bin/bash
# =============================================================================
# Single cluster-level transfer run (zero-shot eval OR fine-tuning).
#
# Transfers a trained LOKI shared policy (20 morphologies) from one task to
# another. Supports both zero-shot evaluation and fine-tuning at a given budget.
#
# Uses LOKI.TRAIN=True (modern mujoco bindings via unimal.py).
# Never uses LOKI.FINETUNE (requires deprecated mujoco_py).
#
# Usage:
#   bash scripts/transfer/run_single_transfer.sh <mode> <src_task> <tgt_task> \
#       <cluster> <seed> <num_clusters> [budget] [num_episodes]
#
# Arguments:
#   mode:         "zero_shot" | "finetune"
#   src_task:     Source task name (e.g., "obstacle", "ft")
#   tgt_task:     Target task name (e.g., "many_obstacle", "bump")
#   cluster:      Cluster index (e.g., 18)
#   seed:         Random seed (e.g., 3429)
#   num_clusters: Total cluster count (e.g., 20 or 40)
#   budget:       Fine-tune budget in env steps (required for finetune mode, e.g., "2e7")
#   num_episodes: Zero-shot eval episodes (default: 50, ignored in finetune mode)
#
# Examples:
#   # Zero-shot: obstacle policy evaluated on many_obstacle
#   bash scripts/transfer/run_single_transfer.sh zero_shot obstacle many_obstacle 18 3429 20
#
#   # Fine-tune: obstacle policy fine-tuned on many_obstacle for 20M steps
#   bash scripts/transfer/run_single_transfer.sh finetune obstacle many_obstacle 18 3429 20 2e7
# =============================================================================

set -e

MODE=$1
SRC_TASK=$2
TGT_TASK=$3
CLUSTER=$4
SEED=$5
NUM_CLUSTERS=$6
BUDGET=${7:-""}
NUM_EPISODES=${8:-50}

NUM_WALKER=20
DROP_FREQ=2
NUM_DROP=2

if [ -z "$MODE" ] || [ -z "$SRC_TASK" ] || [ -z "$TGT_TASK" ] || \
   [ -z "$CLUSTER" ] || [ -z "$SEED" ] || [ -z "$NUM_CLUSTERS" ]; then
    echo "Usage: bash scripts/transfer/run_single_transfer.sh <mode> <src_task> <tgt_task> <cluster> <seed> <num_clusters> [budget] [num_episodes]"
    exit 1
fi

if [ "$MODE" = "finetune" ] && [ -z "$BUDGET" ]; then
    echo "Error: budget is required for finetune mode"
    exit 1
fi

# --- Map task names to output directory names ---
get_task_dir() {
    case "$1" in
        locomotion|ft) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

# --- Map task names to config files ---
get_task_cfg() {
    case "$1" in
        many_obstacle) echo "obstacle" ;;
        locomotion) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

SRC_DIR=$(get_task_dir "$SRC_TASK")
TGT_CFG=$(get_task_cfg "$TGT_TASK")

# --- Source checkpoint path ---
SRC_CKPT_BASE="output/loki/${SRC_DIR}/kmeans_cluster/${NUM_CLUSTERS}/${CLUSTER}/walker${NUM_WALKER}/freq${DROP_FREQ}/drop${NUM_DROP}/seed${SEED}"

# Verify source checkpoint exists
if [ ! -f "metamorph/${SRC_CKPT_BASE}/Unimal-v0.pt" ]; then
    echo "[SKIP] Source checkpoint not found: metamorph/${SRC_CKPT_BASE}/Unimal-v0.pt"
    exit 0
fi

# Find latest xml_step iteration
LATEST_ITER=$(ls "metamorph/${SRC_CKPT_BASE}/xml_step/" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "$LATEST_ITER" ]; then
    echo "[SKIP] No xml_step found in metamorph/${SRC_CKPT_BASE}/xml_step/"
    exit 0
fi

# --- Output paths ---
PAIR="${SRC_TASK}_to_${TGT_TASK}"

if [ "$MODE" = "zero_shot" ]; then
    OUT_DIR="output/transfer/zero_shot/${PAIR}/c${CLUSTER}/seed${SEED}"
    RESULT_MARKER="eval_results.json"
else
    OUT_DIR="output/transfer/finetune/${PAIR}/c${CLUSTER}/steps_${BUDGET}/seed${SEED}"
    RESULT_MARKER="Unimal-v0_results.json"
fi

# Skip if already completed
if [ -f "metamorph/${OUT_DIR}/${RESULT_MARKER}" ]; then
    echo "[SKIP] Already done: ${OUT_DIR}"
    exit 0
fi

# --- Prepare walker directory ---
WALKER_DIR="${OUT_DIR}/walkers"
mkdir -p "metamorph/${WALKER_DIR}/xml"
mkdir -p "metamorph/${WALKER_DIR}/xml_step/0"

for xml_file in "metamorph/${SRC_CKPT_BASE}/xml_step/${LATEST_ITER}"/[0-9]*.xml; do
    [ -f "$xml_file" ] || continue
    cp "$xml_file" "metamorph/${WALKER_DIR}/xml/"
    cp "$xml_file" "metamorph/${WALKER_DIR}/xml_step/0/"
done

XML_COUNT=$(ls "metamorph/${WALKER_DIR}/xml/"*.xml 2>/dev/null | wc -l)
echo "[PREP] ${PAIR} C${CLUSTER}: ${XML_COUNT} XMLs from iter ${LATEST_ITER}"

# --- Target task extra args ---
EXTRA_ARGS=""
if [ "$TGT_TASK" = "many_obstacle" ]; then
    EXTRA_ARGS="OBJECT.NUM_OBSTACLES 150"
fi

# --- Run experiment ---
cd metamorph

if [ "$MODE" = "zero_shot" ]; then
    # ---- Zero-shot evaluation ----
    echo "[EVAL] ${PAIR} C${CLUSTER}: ${NUM_EPISODES} episodes"

    MUJOCO_GL=egl PYTHONPATH=./ python tools/eval_zero_shot.py \
        --cfg "./configs/${TGT_CFG}.yaml" \
        --checkpoint "${SRC_CKPT_BASE}/Unimal-v0.pt" \
        --walker_dir "${WALKER_DIR}" \
        --num_episodes "${NUM_EPISODES}" \
        --out_dir "${OUT_DIR}" \
        $EXTRA_ARGS

    if [ -f "${OUT_DIR}/eval_results.json" ]; then
        MEAN=$(python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_results.json')); print(f'{d[\"mean_reward\"]:.1f}')")
        echo "[DONE] ${PAIR} C${CLUSTER}: mean_reward=${MEAN}"
    else
        echo "[FAIL] ${PAIR} C${CLUSTER}: no eval_results.json"
        exit 1
    fi

else
    # ---- Fine-tuning ----
    # WandB: deterministic run ID for resume support
    export WANDB_PROJECT="LOKI-transfer"
    export WANDB_RUN_ID="transfer-${SRC_TASK}-to-${TGT_TASK}-c${CLUSTER}-steps${BUDGET}-s${SEED}"

    # Log file
    LOG_DIR="../log/transfer/finetune/${PAIR}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/c${CLUSTER}_steps${BUDGET}_seed${SEED}.log"

    echo "[TRAIN] ${PAIR} C${CLUSTER} budget=${BUDGET} wandb=${WANDB_RUN_ID}"

    MUJOCO_GL=egl PYTHONPATH=./ python tools/train_ppo.py \
        --cfg "./configs/${TGT_CFG}.yaml" \
        LOKI.TRAIN True \
        OUT_DIR "${OUT_DIR}" \
        ENV.WALKER_DIR "${WALKER_DIR}" \
        PPO.CHECKPOINT_PATH "${SRC_CKPT_BASE}/Unimal-v0.pt" \
        MODEL.FINETUNE.FULL_MODEL True \
        ENV_TYPE "${TGT_TASK}" \
        PPO.MAX_STATE_ACTION_PAIRS "${BUDGET}" \
        RNG_SEED "${SEED}" \
        $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1

    if [ -f "${OUT_DIR}/Unimal-v0_results.json" ]; then
        echo "[DONE] ${PAIR} C${CLUSTER} budget=${BUDGET}"
    else
        echo "[FAIL] ${PAIR} C${CLUSTER} budget=${BUDGET} — check ${LOG_FILE}"
        exit 1
    fi
fi
