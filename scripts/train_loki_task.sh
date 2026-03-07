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

# Deterministic wandb run ID derived from job parameters, so resumed runs continue
# logging to the same wandb run automatically.
WANDB_RUN_ID="loki-${ENV_TYPE}-c${NUM_CLUSTERS}-idx${CLUSTER_LABEL}-w${NUM_WALKER}-s${RNG_SEED}"
export WANDB_RUN_ID

# Auto-detect partial checkpoint for resume.
# A run is "partial" if xml_step/ has iteration dirs but Unimal-v0_results.json does not
# exist (results.json is only written after training completes fully).
RESUME_ARGS=""
RESUME_ITER=$(ls -d $CKPT_PATH/xml_step/*/ 2>/dev/null | xargs -I{} basename {} | sort -n | tail -1)
RESULTS_FILE="$CKPT_PATH/Unimal-v0_results.json"

if [ -n "$RESUME_ITER" ] && [ ! -f "$RESULTS_FILE" ] && [ -f "$CKPT_PATH/Unimal-v0.pt" ]; then
    # Partial run detected: has checkpointed iterations and model weights, but never finished
    echo "[RESUME] Detected partial run at iter $RESUME_ITER, resuming from checkpoint"
    RESUME_ARGS="PPO.CHECKPOINT_PATH $CKPT_PATH/Unimal-v0.pt LOKI.RESUME_ITER $RESUME_ITER MODEL.FINETUNE.FULL_MODEL True LOKI.INIT_DIR $CKPT_PATH"
fi

# Adaptive CPU thread limits to prevent over-subscription when running concurrently.
# Detects available CPUs and estimated max concurrent jobs, then divides fairly.
TOTAL_CPUS=$(nproc 2>/dev/null || echo 48)
NUM_GPUS_DETECTED=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
EST_MAX_JOBS=$(( NUM_GPUS_DETECTED * 4 ))
[ "$EST_MAX_JOBS" -lt 1 ] && EST_MAX_JOBS=1
THREADS_PER_JOB=$(( TOTAL_CPUS / EST_MAX_JOBS ))
[ "$THREADS_PER_JOB" -lt 2 ] && THREADS_PER_JOB=2

# Note: runs in foreground here; the parent script (train_loki_all_cluster_tasks.sh)
# backgrounds this script so it can properly track the PID for resource scheduling.
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
                        ENV.TYPE "$ENV_TYPE" \
                        VECENV.TYPE DummyVecEnv \
                        RNG_SEED $RNG_SEED \
                        $RESUME_ARGS > ../$LOG_FILE 2>&1
