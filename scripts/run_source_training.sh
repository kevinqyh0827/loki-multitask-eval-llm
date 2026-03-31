#!/bin/bash
# Train a PPO policy from scratch on a source task with a fixed morphology.
# Pre-requisite for Conditions 5 & 6 of the transfer experiments.
#
# This trains a standard PPO policy (no LOKI co-design) using the best
# ft morphology on a different task (e.g., push_box_incline, bump).
# The resulting checkpoint can then be fine-tuned on the target task (incline).
#
# Usage:
#   bash scripts/run_source_training.sh <source_task> <morph_xml_dir> [budget] [seed]
#
# Arguments:
#   source_task:   Task to train on (e.g., "push_box_incline", "bump")
#   morph_xml_dir: Directory containing the source morphology XML file(s)
#                  (e.g., ./output/loki/ft/.../xml_step/1218)
#   budget:        Total state-action pairs (default: 1e8 = 100M)
#   seed:          Random seed (default: 3429)
#
# Example:
#   bash scripts/run_source_training.sh push_box_incline \
#     ./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429/xml_step/1218 \
#     1e8 3429

set -e

SOURCE_TASK=$1
MORPH_XML_DIR=$2
BUDGET=${3:-"1e8"}
SEED=${4:-3429}

if [ -z "$SOURCE_TASK" ] || [ -z "$MORPH_XML_DIR" ]; then
    echo "Usage: bash scripts/run_source_training.sh <source_task> <morph_xml_dir> [budget] [seed]"
    exit 1
fi

OUT_DIR="./output/transfer/source_policies/${SOURCE_TASK}/seed_${SEED}"

# Check if already completed
if [ -f "metamorph/${OUT_DIR}/Unimal-v0_results.json" ]; then
    echo "Source training already completed: ${OUT_DIR}"
    echo "Checkpoint: metamorph/${OUT_DIR}/Unimal-v0.pt"
    exit 0
fi

echo "============================================"
echo "Source Task Training"
echo "Task:    $SOURCE_TASK"
echo "Budget:  $BUDGET"
echo "Seed:    $SEED"
echo "Output:  $OUT_DIR"
echo "============================================"

# Setup walker directory with both xml/ and xml_step/0/ structures
WALKER_PATH="${OUT_DIR}/walkers"
mkdir -p "metamorph/${WALKER_PATH}/xml"
mkdir -p "metamorph/${WALKER_PATH}/xml_step/0"

# Resolve source XML directory (handle paths with or without metamorph/ prefix)
if [ -d "${MORPH_XML_DIR}" ]; then
    SRC_XML_DIR="${MORPH_XML_DIR}"
elif [ -d "metamorph/${MORPH_XML_DIR}" ]; then
    SRC_XML_DIR="metamorph/${MORPH_XML_DIR}"
else
    echo "Error: XML directory not found: ${MORPH_XML_DIR}"
    exit 1
fi

# Copy morphology XML(s)
for xml_file in "${SRC_XML_DIR}"/[0-9]*.xml; do
    if [ -f "$xml_file" ]; then
        cp "$xml_file" "metamorph/${WALKER_PATH}/xml/"
        cp "$xml_file" "metamorph/${WALKER_PATH}/xml_step/0/"
    fi
done
echo "Copied $(ls "metamorph/${WALKER_PATH}/xml/"*.xml 2>/dev/null | wc -l) walker XMLs"

cd metamorph

# Determine config file
if [ "$SOURCE_TASK" = "many_obstacle" ]; then
    CFG_FILE="./configs/obstacle.yaml"
    TASK_ARGS="OBJECT.NUM_OBSTACLES 150 ENV_TYPE many_obstacle"
else
    CFG_FILE="./configs/${SOURCE_TASK}.yaml"
    TASK_ARGS="ENV_TYPE ${SOURCE_TASK}"
fi

# Create log directory
LOG_DIR="../log/transfer/source_training"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SOURCE_TASK}_budget${BUDGET}_seed${SEED}.log"

echo "Starting source policy training on ${SOURCE_TASK}..."
echo "This may take several hours for 100M steps."

# Train PPO from scratch
MUJOCO_GL=egl PYTHONPATH=./ python tools/train_ppo.py \
    --cfg "$CFG_FILE" \
    LOKI.TRAIN True \
    OUT_DIR "${OUT_DIR}" \
    ENV.WALKER_DIR "${WALKER_PATH}" \
    PPO.MAX_STATE_ACTION_PAIRS "${BUDGET}" \
    RNG_SEED "${SEED}" \
    ${TASK_ARGS} \
    > "$LOG_FILE" 2>&1

echo ""
echo "Source training completed!"
echo "Checkpoint: metamorph/${OUT_DIR}/Unimal-v0.pt"
echo "Results:    metamorph/${OUT_DIR}/Unimal-v0_results.json"
echo "Log:        $LOG_FILE"
