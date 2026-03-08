#!/bin/bash
#SBATCH --job-name=loki-resume32
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-resume32-%j.out
#SBATCH --error=log/slurm/loki-resume32-%j.err

# === Resume the 32 incomplete evaluations ===
# Locomotion clusters 0-19 (20 runs) + Obstacle clusters 0-11 (12 runs)
# These runs have partial checkpoints and will auto-resume from last iteration.
#
# Submit: sbatch --gres=gpu:h200:8 scripts/run_resume_32.sh

# Setup environment
module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki

# Navigate to repo
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

# Create log directories
mkdir -p log/slurm
mkdir -p log/train_loki_task/locomotion
mkdir -p log/train_loki_task/obstacle

# Print job and node info
echo "=== LOKI Resume 32 Incomplete Evaluations ==="
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
bash scripts/schedule_resume_32.sh $NUM_GPUS 0 ${TEST_MODE:-0}

echo ""
echo "End time: $(date)"
