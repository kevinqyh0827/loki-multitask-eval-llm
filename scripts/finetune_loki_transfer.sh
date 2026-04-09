#!/bin/bash
# Fine-tune a LOKI cluster policy from any source task to any target task.
#
# Usage:
#   bash scripts/finetune_loki_transfer.sh <num_walkers> <num_clusters> <cluster_idx> <seed> <source_task> <target_task> [max_steps]
#
# Examples:
#   # Transfer obstacle -> many_obstacle (full 1e8 budget)
#   bash scripts/finetune_loki_transfer.sh 20 40 18 3429 obstacle many_obstacle
#
#   # Transfer obstacle -> bump (20M steps budget)
#   bash scripts/finetune_loki_transfer.sh 20 40 18 3429 obstacle bump 2e7
#
#   # Zero-shot eval only (0 fine-tuning steps)
#   bash scripts/finetune_loki_transfer.sh 20 40 18 3429 obstacle many_obstacle 0

NUM_WALKER=$1
NUM_CLUSTERS=$2
CLUSTER_LABEL=$3
RNG_SEED=$4
SOURCE_TASK=$5
TARGET_TASK=$6
MAX_STEPS=${7:-1e8}

DROP_FREQ=2
NUM_DROP=2

# Map task names to config directory names
declare -A TASK_TO_DIR
TASK_TO_DIR[ft]=ft
TASK_TO_DIR[locomotion]=ft
TASK_TO_DIR[obstacle]=obstacle
TASK_TO_DIR[many_obstacle]=obstacle  # uses obstacle.yaml with NUM_OBSTACLES override
TASK_TO_DIR[bump]=bump
TASK_TO_DIR[incline]=incline
TASK_TO_DIR[push_box_incline]=push_box_incline
TASK_TO_DIR[manipulation_ball]=manipulation_ball
TASK_TO_DIR[exploration]=exploration
TASK_TO_DIR[patrol]=patrol

# Map source task name to output directory name
declare -A TASK_TO_OUTDIR
TASK_TO_OUTDIR[ft]=ft
TASK_TO_OUTDIR[locomotion]=ft
TASK_TO_OUTDIR[obstacle]=obstacle
TASK_TO_OUTDIR[many_obstacle]=many_obstacle
TASK_TO_OUTDIR[bump]=bump
TASK_TO_OUTDIR[incline]=incline
TASK_TO_OUTDIR[push_box_incline]=push_box_incline
TASK_TO_OUTDIR[manipulation_ball]=manipulation_ball
TASK_TO_OUTDIR[exploration]=exploration
TASK_TO_OUTDIR[patrol]=patrol

SOURCE_OUTDIR=${TASK_TO_OUTDIR[$SOURCE_TASK]}
TARGET_CONFIG=${TASK_TO_DIR[$TARGET_TASK]}

LOG_PATH="./log/finetune_transfer/${SOURCE_TASK}_to_${TARGET_TASK}"
mkdir -p "$LOG_PATH"

LOG_FILE="$LOG_PATH/cluster${NUM_CLUSTERS}_idx${CLUSTER_LABEL}_walker${NUM_WALKER}_seed${RNG_SEED}_steps${MAX_STEPS}.log"

# Source checkpoint and XMLs
CKPT_PATH="./output/loki/${SOURCE_OUTDIR}/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED"
SAVE_PATH="./output/loki_transfer/${SOURCE_TASK}_to_${TARGET_TASK}/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED/steps_${MAX_STEPS}"
XML_PATH="$CKPT_PATH/xml_step/1218"

# Copy morphology XMLs from source task's final elite pool
WALKER_PATH="$CKPT_PATH/xml_step"
mkdir -p "metamorph/$WALKER_PATH/xml"
cp "metamorph/$XML_PATH"/[0-9]*.xml "metamorph/$WALKER_PATH/xml"

# Convert xml to pkl (needed for observation construction)
PYTHONPATH=. python tools/xml_2_pkl.py --input_dir "metamorph/$WALKER_PATH"

cd metamorph

# Build extra config overrides
EXTRA_OPTS=""
if [ "$TARGET_TASK" = "many_obstacle" ]; then
    EXTRA_OPTS="OBJECT.NUM_OBSTACLES 150"
fi

echo "============================================================"
echo "  Transfer: ${SOURCE_TASK} -> ${TARGET_TASK}"
echo "  Cluster: ${CLUSTER_LABEL}/${NUM_CLUSTERS}, Walker: ${NUM_WALKER}, Seed: ${RNG_SEED}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Output: ${SAVE_PATH}"
echo "============================================================"

PYTHONPATH=./ python tools/train_ppo.py --cfg "./configs/${TARGET_CONFIG}.yaml" \
    LOKI.FINETUNE True \
    OUT_DIR "$SAVE_PATH" \
    ENV.WALKER_DIR "$WALKER_PATH" \
    PPO.CHECKPOINT_PATH "$CKPT_PATH/Unimal-v0.pt" \
    MODEL.FINETUNE.FULL_MODEL True \
    ENV_TYPE "$TARGET_TASK" \
    PPO.MAX_STATE_ACTION_PAIRS "$MAX_STEPS" \
    $EXTRA_OPTS \
    > "../$LOG_FILE" 2>&1 &

echo "PID: $!"
echo "Log: $LOG_FILE"
