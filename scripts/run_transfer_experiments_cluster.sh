#!/bin/bash
# Transfer learning experiments for HPC/SLURM cluster.
#
# Runs all transfer experiment conditions with GPU-aware scheduling.
# Follows the same two-phase scheduling pattern as train_loki_all_cluster_tasks.sh.
#
# SLURM submission:
#   sbatch --gres=gpu:a100:2 --cpus-per-task=32 --mem=128G --time=2-00:00:00 \
#          --output=log/slurm/transfer-%j.out --error=log/slurm/transfer-%j.err \
#          scripts/run_transfer_experiments_cluster.sh
#
# Direct execution (interactive node):
#   bash scripts/run_transfer_experiments_cluster.sh [num_gpus] [max_concurrent_per_gpu]
#
# GPU recommendations (each fine-tune job uses ~2-3 GB VRAM, source training ~8-10 GB):
#
#   GPUs requested    Phase 0 (source)    Phase 2 (90 jobs)    Total
#   ─────────────     ────────────────    ─────────────────    ─────
#   1x H200-141GB     ~20 hrs (sequential) ~3 hrs              ~23 hrs
#   2x H200-141GB     ~10 hrs (parallel)   ~1.5 hrs            ~12 hrs  ← recommended
#   1x A100-80GB      ~20 hrs              ~3 hrs              ~23 hrs
#   2x A100-80GB      ~10 hrs (parallel)   ~1.5 hrs            ~12 hrs
#
# With 2 GPUs, Phase 0 runs both source tasks IN PARALLEL (one per GPU),
# cutting source training time in half. Phase 2 distributes 90 small jobs
# across both GPUs with round-robin scheduling.

#SBATCH --job-name=loki-transfer
#SBATCH --partition=work1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=log/slurm/transfer-%j.out
#SBATCH --error=log/slurm/transfer-%j.err

set -e

# ==================== Configuration ====================

NUM_GPUS=${1:-0}  # 0 = auto-detect
MAX_CONCURRENT_OVERRIDE=${2:-0}  # 0 = auto-detect from VRAM tier

# Source: best ft morphology (cluster 18, agent 0)
FT_SOURCE_DIR="./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429"
MORPH_XML_DIR="${FT_SOURCE_DIR}/xml_step/1218"

TARGET="incline"
BUDGETS=("1e6" "2e6" "5e6" "1e7" "2e7" "5e7")
SEEDS=(3429 1409 42)

PBI_SOURCE_DIR="./output/transfer/source_policies/push_box_incline/seed_3429"
BUMP_SOURCE_DIR="./output/transfer/source_policies/bump/seed_3429"

STAGGER_TIME=30   # 30s between launches (shorter than LOKI training since these are lighter)
RAM_THRESHOLD=40000  # 40 GB free RAM minimum (lower than LOKI since single-morph PPO uses less)

# ==================== Cluster Environment Setup ====================

# Load modules if on SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running as SLURM job $SLURM_JOB_ID on $(hostname)"
    module load cuda/12.3 2>/dev/null || true
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate loki 2>/dev/null || true
fi

export MUJOCO_GL=egl

# ==================== GPU Detection ====================

if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
fi

get_gpu_total_mem() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$1 2>/dev/null | tr -d ' '
}

get_gpu_free_mem() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$1 2>/dev/null | tr -d ' '
}

get_free_ram() {
    free -m 2>/dev/null | awk '/^Mem:/ {print $7}'
}

detect_max_concurrent_per_gpu() {
    local gpu_mem=$(get_gpu_total_mem 0)
    if [ "$gpu_mem" -le 35000 ]; then
        # RTX 5090 / RTX 4090 (~32 GB) — single-morph PPO uses ~2-3 GB each
        MAX_CONCURRENT_PER_GPU=4
    elif [ "$gpu_mem" -le 50000 ]; then
        # A100-40GB
        MAX_CONCURRENT_PER_GPU=6
    elif [ "$gpu_mem" -le 100000 ]; then
        # A100-80GB
        MAX_CONCURRENT_PER_GPU=10
    else
        # H200 / larger
        MAX_CONCURRENT_PER_GPU=16
    fi
    echo "[CONFIG] GPU 0 has ${gpu_mem} MiB VRAM -> max ${MAX_CONCURRENT_PER_GPU} concurrent per GPU"
}

if [ "$MAX_CONCURRENT_OVERRIDE" -gt 0 ]; then
    MAX_CONCURRENT_PER_GPU=$MAX_CONCURRENT_OVERRIDE
else
    detect_max_concurrent_per_gpu
fi

TOTAL_MAX=$((NUM_GPUS * MAX_CONCURRENT_PER_GPU))

echo "================================================================"
echo "Transfer Learning Experiments — Cluster Mode"
echo "================================================================"
echo "GPUs: $NUM_GPUS, Max concurrent per GPU: $MAX_CONCURRENT_PER_GPU, Total max: $TOTAL_MAX"
echo "Target task: $TARGET"
echo "Budgets: ${BUDGETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "================================================================"

# ==================== Helper Functions ====================

NEXT_GPU=0
declare -A GPU_JOB_COUNT
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_JOB_COUNT[$gpu]=0
done

RUNNING_PIDS=()

advance_gpu_rr() {
    TARGET_GPU=$((NEXT_GPU % NUM_GPUS))
    NEXT_GPU=$(( (NEXT_GPU + 1) % NUM_GPUS ))
}

update_running() {
    local new_pids=()
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        fi
    done
    RUNNING_PIDS=("${new_pids[@]}")
}

wait_for_slot() {
    while true; do
        update_running
        if [ "${#RUNNING_PIDS[@]}" -lt "$TOTAL_MAX" ]; then
            # Also check RAM
            local free_ram=$(get_free_ram)
            if [ "$free_ram" -gt "$RAM_THRESHOLD" ]; then
                break
            fi
            echo "[WAIT] Low RAM: ${free_ram} MiB free < ${RAM_THRESHOLD} MiB threshold"
        fi
        sleep 10
    done
}

launch_job() {
    local mode=$1
    local target=$2
    local budget=$3
    local seed=$4
    local morph_dir=$5
    local ckpt=${6:-""}
    local model=${7:-"ActorCritic"}
    local source_name=${8:-""}

    wait_for_slot
    advance_gpu_rr

    echo "[LAUNCH] GPU $TARGET_GPU: $mode $target budget=$budget seed=$seed source=$source_name"

    CUDA_VISIBLE_DEVICES=$TARGET_GPU \
    bash scripts/run_finetune_single.sh "$mode" "$target" "$budget" "$seed" \
        "$morph_dir" "$ckpt" "$model" "$source_name" &

    local pid=$!
    RUNNING_PIDS+=($pid)
    GPU_JOB_COUNT[$TARGET_GPU]=$(( ${GPU_JOB_COUNT[$TARGET_GPU]} + 1 ))

    sleep $STAGGER_TIME
}

# ==================== Phase 0: Source Policy Training ====================
echo ""
echo "================================================================"
echo "PHASE 0: Source policy training"
echo "================================================================"

# With multiple GPUs, run both source training jobs in PARALLEL on separate GPUs.
# Each job uses ~8-10 GB VRAM, so even A100-40GB can handle one per GPU.
# With 1 GPU, they run sequentially.

PHASE0_PIDS=()

if [ ! -f "metamorph/${PBI_SOURCE_DIR}/Unimal-v0_results.json" ]; then
    echo "Training push_box_incline source policy (100M steps) on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_source_training.sh push_box_incline \
        "$MORPH_XML_DIR" 1e8 3429 &
    PHASE0_PIDS+=($!)
else
    echo "push_box_incline source policy already exists."
fi

if [ ! -f "metamorph/${BUMP_SOURCE_DIR}/Unimal-v0_results.json" ]; then
    # Use GPU 1 if available, otherwise GPU 0 (will run after push_box_incline finishes)
    BUMP_GPU=$(( NUM_GPUS > 1 ? 1 : 0 ))
    echo "Training bump source policy (100M steps) on GPU ${BUMP_GPU}..."
    CUDA_VISIBLE_DEVICES=$BUMP_GPU bash scripts/run_source_training.sh bump \
        "$MORPH_XML_DIR" 1e8 3429 &
    PHASE0_PIDS+=($!)
else
    echo "bump source policy already exists."
fi

# Wait for all Phase 0 jobs to complete
if [ ${#PHASE0_PIDS[@]} -gt 0 ]; then
    echo "Waiting for ${#PHASE0_PIDS[@]} source training job(s)..."
    for pid in "${PHASE0_PIDS[@]}"; do
        wait "$pid"
    done
fi

echo "Phase 0 complete."

# ==================== Phase 1: Zero-shot Evaluation ====================
echo ""
echo "================================================================"
echo "PHASE 1: Zero-shot evaluation"
echo "================================================================"

if [ ! -f "output/transfer/zero_shot/${TARGET}/eval_results.json" ]; then
    CUDA_VISIBLE_DEVICES=0 bash scripts/eval_zero_shot.sh "$FT_SOURCE_DIR" "$TARGET" 50
else
    echo "Zero-shot eval already exists."
    python3 -c "import json; d=json.load(open('output/transfer/zero_shot/${TARGET}/eval_results.json')); print(f'  Mean reward: {d[\"mean_reward\"]:.1f}')"
fi

echo "Phase 1 complete."

# ==================== Phase 2: Budget Sweep ====================
echo ""
echo "================================================================"
echo "PHASE 2: Fine-tune and scratch conditions"
echo "================================================================"

TOTAL_LAUNCHED=0

for BUDGET in "${BUDGETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Condition 2: Fine-tune from ft (LOKI DLS)
        launch_job finetune "$TARGET" "$BUDGET" "$SEED" \
            "$MORPH_XML_DIR" \
            "${FT_SOURCE_DIR}/Unimal-v0.pt" \
            ActorCritic ft
        TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

        # Condition 3: Scratch Transformer
        launch_job scratch "$TARGET" "$BUDGET" "$SEED" \
            "$MORPH_XML_DIR"
        TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

        # Condition 4: Scratch MLP
        launch_job scratch_mlp "$TARGET" "$BUDGET" "$SEED" \
            "$MORPH_XML_DIR"
        TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

        # Condition 5: Fine-tune from push_box_incline
        launch_job finetune "$TARGET" "$BUDGET" "$SEED" \
            "$MORPH_XML_DIR" \
            "metamorph/${PBI_SOURCE_DIR}/Unimal-v0.pt" \
            ActorCritic push_box_incline
        TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

        # Condition 6: Fine-tune from bump
        launch_job finetune "$TARGET" "$BUDGET" "$SEED" \
            "$MORPH_XML_DIR" \
            "metamorph/${BUMP_SOURCE_DIR}/Unimal-v0.pt" \
            ActorCritic bump
        TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))
    done
done

echo ""
echo "All $TOTAL_LAUNCHED jobs launched. Waiting for completion..."
wait

echo ""
echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================"
echo "Total Phase 2 jobs: $TOTAL_LAUNCHED"
echo ""
echo "Run analysis:"
echo "  python tools/analyze_transfer_results.py --base_dir metamorph/output/transfer"
echo "  python tools/plot_transfer_curves.py --base_dir metamorph/output/transfer"
