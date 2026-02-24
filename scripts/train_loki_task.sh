#!/bin/bash
# Train LOKI co-design from scratch for a specific task and cluster.
# Usage: bash scripts/train_loki_task.sh <num_walkers> <num_clusters> <cluster_label> <seed> <task_name> [gpu_id]
# Example: bash scripts/train_loki_task.sh 20 20 0 3429 obstacle 0
#
# Supported tasks: locomotion, obstacle, incline, bump, push_box_incline

NUM_WALKER=$1
NUM_CLUSTERS=$2
CLUSTER_LABEL=$3
RNG_SEED=$4
TASK_NAME=$5
GPU_ID=${6:-0}

DROP_FREQ=2
NUM_DROP=2
NUM_SAMPLE=128

# Map task name to config YAML and output dir name
case "$TASK_NAME" in
    locomotion)
        CFG_FILE="./configs/ft.yaml"
        ENV_TYPE="ft"
        ;;
    obstacle)
        CFG_FILE="./configs/obstacle.yaml"
        ENV_TYPE="obstacle"
        ;;
    incline)
        CFG_FILE="./configs/incline.yaml"
        ENV_TYPE="incline"
        ;;
    bump)
        CFG_FILE="./configs/bump.yaml"
        ENV_TYPE="bump"
        ;;
    push_box_incline)
        CFG_FILE="./configs/push_box_incline.yaml"
        ENV_TYPE="push_box_incline"
        ;;
    *)
        echo "Error: Unknown task '$TASK_NAME'. Supported: locomotion, obstacle, incline, bump, push_box_incline"
        exit 1
        ;;
esac

LOG_PATH="./log/train_loki_task/$TASK_NAME"
mkdir -p $LOG_PATH

LOG_FILE="$LOG_PATH/cluster${NUM_CLUSTERS}_idx${CLUSTER_LABEL}_walker${NUM_WALKER}_freq${DROP_FREQ}_drop${NUM_DROP}_seed$RNG_SEED.log"

CKPT_PATH="./output/loki/$ENV_TYPE/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED"

VAE_PATH="VAE_50k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200_20260209_215748"

cd metamorph
CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=./ python tools/train_loki.py \
                        --cfg $CFG_FILE \
                        --vae_path $VAE_PATH \
                        --device cuda:0 \
                        OUT_DIR $CKPT_PATH \
                        LOKI.TRAIN True \
                        LOKI.NUM_WALKER $NUM_WALKER \
                        LOKI.NUM_DROP_WALKER $NUM_DROP \
                        LOKI.DROP_FREQ $DROP_FREQ \
                        PPO.MAX_STATE_ACTION_PAIRS 1e8 \
                        LOG_PERIOD 10 \
                        LOKI.NUM_CLUSTERS $NUM_CLUSTERS \
                        LOKI.CLUSTER_LABEL $CLUSTER_LABEL \
                        LOKI.DROP_WARMUP $DROP_FREQ \
                        LOKI.SAMPLE_SIZE $NUM_SAMPLE \
                        LOKI.MUTATE_SAMPLE False \
                        ENV.TYPE "$ENV_TYPE" \
                        RNG_SEED $RNG_SEED > ../$LOG_FILE 2>&1 &

echo "Launched: task=$TASK_NAME cluster=$CLUSTER_LABEL (PID=$!) -> log: $LOG_FILE"
