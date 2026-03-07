#!/bin/bash
#SBATCH --job-name=loki-remaining28
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-remaining28-%j.out
#SBATCH --error=log/slurm/loki-remaining28-%j.err

# === Launch only the 28 remaining NEW evaluations ===
# Obstacle clusters 12-19 (8 runs) + Incline clusters 0-19 (20 runs)
# Skips the 32 incomplete runs (locomotion 0-19, obstacle 0-11) for now.
#
# Submit: sbatch --gres=gpu:h200:8 scripts/run_remaining_28.sh

# Setup environment
module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki

# Navigate to repo
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

# Create log directories
mkdir -p log/slurm
mkdir -p log/train_loki_task/obstacle
mkdir -p log/train_loki_task/incline

# Print job and node info
echo "=== LOKI Remaining 28 Evaluations ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# Run the targeted scheduler
# Args: num_gpus, max_concurrent_per_gpu (0=auto-detect), test_mode (0=full, 1=reduced steps)
bash scripts/schedule_remaining_28.sh $NUM_GPUS 0 ${TEST_MODE:-0}

echo ""
echo "End time: $(date)"
