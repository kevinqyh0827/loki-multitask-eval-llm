#!/bin/bash
#SBATCH --job-name=loki-eval
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-eval-%x-%j.out
#SBATCH --error=log/slurm/loki-eval-%x-%j.err

# === LOKI Single-Task Evaluation (Backup Plan) ===
# Runs 20 clusters for a single task on a single GPU.
# Submit 3 separate jobs (one per task) for full coverage.
#
# Usage as SLURM job:
#   sbatch --gres=gpu:h100:1 --job-name=loki-locomotion scripts/run_eval_single_task.sh locomotion
#   sbatch --gres=gpu:h100:1 --job-name=loki-obstacle   scripts/run_eval_single_task.sh obstacle
#   sbatch --gres=gpu:h100:1 --job-name=loki-incline    scripts/run_eval_single_task.sh incline
#
# With test mode (1e5 steps):
#   TEST_MODE=1 sbatch --gres=gpu:h100:1 --job-name=loki-locomotion scripts/run_eval_single_task.sh locomotion
#
# Submit all 3 at once:
#   for task in locomotion obstacle incline; do
#       sbatch --gres=gpu:h100:1 --mem=256G --job-name=loki-${task} scripts/run_eval_single_task.sh $task
#   done

TASK_NAME=${1:?  "Error: task name required. Usage: sbatch ... scripts/run_eval_single_task.sh <locomotion|obstacle|incline>"}

# Setup environment
module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki

# Navigate to repo
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

# Create log directories
mkdir -p log/slurm
mkdir -p log/train_loki_task/$TASK_NAME

# Print job and node info
echo "=== LOKI Single-Task Evaluation Job ==="
echo "Task: $TASK_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

echo "--- GPU Details ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""
nvidia-smi
echo ""

echo "--- System RAM ---"
free -h
echo ""

# Run single-task scheduler
# Args: task_name, max_concurrent (0=auto-detect), test_mode (0=full, 1=reduced steps)
bash scripts/train_loki_single_task.sh $TASK_NAME 0 ${TEST_MODE:-0}

echo ""
echo "End time: $(date)"
echo ""

ENV_TYPE=$TASK_NAME
if [ "$TASK_NAME" = "locomotion" ]; then
    ENV_TYPE="ft"
fi
echo "Check results with:"
echo "  ls metamorph/output/loki/$ENV_TYPE/kmeans_cluster/20/*/walker20/freq2/drop2/seed3429/Unimal-v0.pt"
