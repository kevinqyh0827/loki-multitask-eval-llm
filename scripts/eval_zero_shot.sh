#!/bin/bash
# Zero-shot policy evaluation: evaluate a trained policy on a new task without training.
#
# Usage:
#   bash scripts/eval_zero_shot.sh <source_ckpt_dir> <target_task> [num_episodes]
#
# Arguments:
#   source_ckpt_dir: Path to source training output dir (contains Unimal-v0.pt and xml_step/)
#                    e.g., ./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429
#   target_task:     Target task name (e.g., incline, obstacle, bump)
#   num_episodes:    Number of evaluation episodes (default: 50)
#
# Example:
#   bash scripts/eval_zero_shot.sh \
#     ./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429 \
#     incline 50

set -e

SOURCE_DIR=$1
TARGET_TASK=$2
NUM_EPISODES=${3:-50}

if [ -z "$SOURCE_DIR" ] || [ -z "$TARGET_TASK" ]; then
    echo "Usage: bash scripts/eval_zero_shot.sh <source_ckpt_dir> <target_task> [num_episodes]"
    exit 1
fi

# Determine the latest xml_step iteration
LATEST_ITER=$(ls "metamorph/$SOURCE_DIR/xml_step/" | sort -n | tail -1)
echo "Using xml_step iteration: $LATEST_ITER"

# Setup walker directory with xml_step/0/ structure
# (unimal.UnimalEnv expects XMLs in {WALKER_DIR}/xml_step/0/)
# Copy ALL agent XMLs (needed for MultiEnvWrapper observation space match)
EVAL_OUT="./output/transfer/zero_shot/${TARGET_TASK}"
WALKER_PATH="${EVAL_OUT}/walkers"
mkdir -p "metamorph/${WALKER_PATH}/xml_step/0"

# Copy all numbered agent XMLs (not tmp_ files)
for xml_file in "metamorph/${SOURCE_DIR}/xml_step/${LATEST_ITER}"/[0-9]*.xml; do
    if [ -f "$xml_file" ]; then
        cp "$xml_file" "metamorph/${WALKER_PATH}/xml_step/0/"
    fi
done

echo "Copied $(ls "metamorph/${WALKER_PATH}/xml_step/0/"*.xml 2>/dev/null | wc -l) agent XMLs"

# Determine config file
if [ "$TARGET_TASK" = "many_obstacle" ]; then
    CFG_FILE="./configs/obstacle.yaml"
    EXTRA_ARGS="OBJECT.NUM_OBSTACLES 150 ENV_TYPE many_obstacle"
else
    CFG_FILE="./configs/${TARGET_TASK}.yaml"
    EXTRA_ARGS=""
fi

# Run zero-shot evaluation
cd metamorph
MUJOCO_GL=egl PYTHONPATH=./ python tools/eval_zero_shot.py \
    --cfg "$CFG_FILE" \
    --checkpoint "${SOURCE_DIR}/Unimal-v0.pt" \
    --walker_dir "${WALKER_PATH}" \
    --num_episodes $NUM_EPISODES \
    --out_dir "../${EVAL_OUT}" \
    $EXTRA_ARGS

echo ""
echo "Done! Results saved to ${EVAL_OUT}/eval_results.json"
