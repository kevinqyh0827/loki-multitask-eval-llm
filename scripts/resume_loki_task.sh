#!/bin/bash
# Resume a crashed/stopped LOKI training run from checkpoint.
# Usage: bash scripts/resume_loki_task.sh <num_walkers> <num_clusters> <cluster_label> <seed> <task_name> [wandb_run_id]
# Example: bash scripts/resume_loki_task.sh 20 20 2 3429 incline 9mua0w8t
#
# The script automatically finds the latest checkpoint iteration from xml_step/.
# Optionally pass wandb_run_id to continue logging to the same wandb run.

NUM_WALKER=$1
NUM_CLUSTERS=$2
CLUSTER_LABEL=$3
RNG_SEED=$4
TASK_NAME=$5
WANDB_RUN_ID=$6

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

CKPT_PATH="./output/loki/$ENV_TYPE/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED"

# Find the latest checkpoint iteration (highest numbered directory in xml_step/)
RESUME_ITER=$(ls -d metamorph/$CKPT_PATH/xml_step/*/ 2>/dev/null | xargs -I{} basename {} | sort -n | tail -1)

if [ -z "$RESUME_ITER" ]; then
    echo "Error: No checkpoint found in metamorph/$CKPT_PATH/xml_step/"
    exit 1
fi

# Verify the model checkpoint exists
MODEL_CKPT="metamorph/$CKPT_PATH/Unimal-v0.pt"
if [ ! -f "$MODEL_CKPT" ]; then
    echo "Error: Model checkpoint not found at $MODEL_CKPT"
    exit 1
fi

echo "Resuming from iteration $RESUME_ITER"
echo "  Model checkpoint: $MODEL_CKPT"
echo "  Output dir: $CKPT_PATH"

LOG_PATH="./log/train_loki_task/$TASK_NAME"
mkdir -p $LOG_PATH

LOG_FILE="$LOG_PATH/cluster${NUM_CLUSTERS}_idx${CLUSTER_LABEL}_walker${NUM_WALKER}_freq${DROP_FREQ}_drop${NUM_DROP}_seed$RNG_SEED.log"

VAE_PATH="VAE_50k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200_20260209_215748"

# Set wandb run ID for resume if provided
WANDB_ENV=""
if [ -n "$WANDB_RUN_ID" ]; then
    WANDB_ENV="WANDB_RUN_ID=$WANDB_RUN_ID"
    echo "  Resuming wandb run: $WANDB_RUN_ID"
fi

cd metamorph
env $WANDB_ENV PYTHONPATH=./ python tools/train_loki.py \
                        --cfg $CFG_FILE \
                        --vae_path $VAE_PATH \
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
                        RNG_SEED $RNG_SEED \
                        PPO.CHECKPOINT_PATH $CKPT_PATH/Unimal-v0.pt \
                        LOKI.RESUME_ITER $RESUME_ITER \
                        MODEL.FINETUNE.FULL_MODEL True >> ../$LOG_FILE 2>&1 &

echo "Launched resume: task=$TASK_NAME cluster=$CLUSTER_LABEL iter=$RESUME_ITER (PID=$!) -> log: $LOG_FILE"
