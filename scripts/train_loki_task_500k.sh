#!/bin/bash
# Train LOKI co-design for a specific task and cluster using 500K data.
# Usage: bash scripts/train_loki_task_500k.sh <num_walkers> <num_clusters> <cluster_label> <seed> <task_name> [gpu_id] [test_mode]
# Example: bash scripts/train_loki_task_500k.sh 20 40 0 3429 obstacle 0
#
# Differences from train_loki_task.sh (50K):
#   - Uses 500K VAE checkpoint (VAE_500k_hdim32_...)
#   - Reads cluster tars from data_500k/ via LOKI.CLUSTER_DATA_DIR
#   - Logs to wandb project "LOKI-500k" (override with WANDB_PROJECT env var)
#   - Outputs to output/loki_500k/<task>/...
#   - Supports all 9 tasks
#
# Supported tasks: locomotion, obstacle, many_obstacle, bump, incline, push_box_incline, exploration, patrol, manipulation_ball

NUM_WALKER=$1
NUM_CLUSTERS=$2
CLUSTER_LABEL=$3
RNG_SEED=$4
TASK_NAME=$5
GPU_ID=${6:-0}
TEST_MODE=${7:-0}                   # 1 = reduced steps for testing (1e5 instead of 1e8)

DROP_FREQ=2
NUM_DROP=2
NUM_SAMPLE=128

# Set training steps based on test mode
if [ "$TEST_MODE" -eq 1 ]; then
    MAX_STEPS="1e5"
else
    MAX_STEPS="1e8"
fi

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
    many_obstacle)
        CFG_FILE="./configs/many_obstacle.yaml"
        ENV_TYPE="many_obstacle"
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
    exploration)
        CFG_FILE="./configs/exploration.yaml"
        ENV_TYPE="exploration"
        ;;
    patrol)
        CFG_FILE="./configs/patrol.yaml"
        ENV_TYPE="patrol"
        ;;
    manipulation_ball)
        CFG_FILE="./configs/manipulation_ball.yaml"
        ENV_TYPE="manipulation_ball"
        ;;
    *)
        echo "Error: Unknown task '$TASK_NAME'. Supported: locomotion, obstacle, many_obstacle, bump, incline, push_box_incline, exploration, patrol, manipulation_ball"
        exit 1
        ;;
esac

LOG_PATH="./log/train_loki_task_500k/$TASK_NAME"
mkdir -p $LOG_PATH

LOG_FILE="$LOG_PATH/cluster${NUM_CLUSTERS}_idx${CLUSTER_LABEL}_walker${NUM_WALKER}_freq${DROP_FREQ}_drop${NUM_DROP}_seed$RNG_SEED.log"

# Output under loki_500k/ to separate from 50K results
CKPT_PATH="./output/loki_500k/$ENV_TYPE/kmeans_cluster/$NUM_CLUSTERS/$CLUSTER_LABEL/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED"

# ========== 500K-specific paths ==========
# VAE checkpoint trained on 500K data (update the timestamp once training finishes)
VAE_PATH="${VAE_500K_CKPT:-VAE_500k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200}"

# Cluster data directory for 500K
CLUSTER_DATA_DIR="data_500k"

cd metamorph

# Deterministic wandb run ID (includes "500k" to avoid collision with 50K runs)
WANDB_RUN_ID="loki500k-${ENV_TYPE}-c${NUM_CLUSTERS}-idx${CLUSTER_LABEL}-w${NUM_WALKER}-s${RNG_SEED}"
export WANDB_RUN_ID
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-LOKI-500k}"
echo "[WANDB] Mode: $WANDB_MODE | Project: $WANDB_PROJECT | Run ID: $WANDB_RUN_ID"

# Quick wandb API connectivity check (non-blocking, 5s timeout)
if [ "$WANDB_MODE" = "online" ]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 https://api.wandb.ai/healthcheck 2>/dev/null)
    if [ -n "$HTTP_CODE" ] && [ "$HTTP_CODE" != "000" ]; then
        echo "[WANDB] API reachable (HTTP $HTTP_CODE) — online logging enabled"
    else
        echo "[WANDB] WARNING: API unreachable — will fall back to offline if init fails"
    fi
fi

# Auto-detect partial checkpoint for resume.
RESUME_ARGS=""
RESUME_ITER=$(ls -d $CKPT_PATH/xml_step/*/ 2>/dev/null | xargs -I{} basename {} | sort -n | tail -1)
RESULTS_FILE="$CKPT_PATH/Unimal-v0_results.json"

if [ -n "$RESUME_ITER" ] && [ ! -f "$RESULTS_FILE" ] && [ -f "$CKPT_PATH/Unimal-v0.pt" ]; then
    echo "[RESUME] Detected partial run at iter $RESUME_ITER, resuming from checkpoint"
    RESUME_ARGS="PPO.CHECKPOINT_PATH $CKPT_PATH/Unimal-v0.pt LOKI.RESUME_ITER $RESUME_ITER MODEL.FINETUNE.FULL_MODEL True LOKI.INIT_DIR $CKPT_PATH"
fi

# Adaptive CPU thread limits
TOTAL_CPUS=$(nproc 2>/dev/null || echo 48)
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
EST_MAX_JOBS=$(( NUM_GPUS_DETECTED * 4 ))
[ "$EST_MAX_JOBS" -lt 1 ] && EST_MAX_JOBS=1
THREADS_PER_JOB=$(( TOTAL_CPUS / EST_MAX_JOBS ))
[ "$THREADS_PER_JOB" -lt 2 ] && THREADS_PER_JOB=2

OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    LOKI_SEQUENTIAL_SIMILARITY=1 \
    CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=./ python tools/train_loki.py \
                        --cfg $CFG_FILE \
                        --vae_path $VAE_PATH \
                        --device cuda:0 \
                        OUT_DIR $CKPT_PATH \
                        LOKI.TRAIN True \
                        LOKI.NUM_WALKER $NUM_WALKER \
                        LOKI.NUM_DROP_WALKER $NUM_DROP \
                        LOKI.DROP_FREQ $DROP_FREQ \
                        PPO.MAX_STATE_ACTION_PAIRS $MAX_STEPS \
                        LOG_PERIOD 10 \
                        LOKI.NUM_CLUSTERS $NUM_CLUSTERS \
                        LOKI.CLUSTER_LABEL $CLUSTER_LABEL \
                        LOKI.DROP_WARMUP $DROP_FREQ \
                        LOKI.SAMPLE_SIZE $NUM_SAMPLE \
                        LOKI.MUTATE_SAMPLE False \
                        LOKI.CLUSTER_DATA_DIR $CLUSTER_DATA_DIR \
                        ENV.TYPE "$ENV_TYPE" \
                        VECENV.TYPE DummyVecEnv \
                        RNG_SEED $RNG_SEED \
                        $RESUME_ARGS > ../$LOG_FILE 2>&1
