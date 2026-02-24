#!/bin/bash
#SBATCH --job-name=loki-eval
#SBATCH --partition=work1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-eval-%j.out
#SBATCH --error=log/slurm/loki-eval-%j.err

# === LOKI Cluster-Task Evaluation on Palmetto ===
# Runs 3 tasks (locomotion, obstacle, incline) x 20 clusters = 60 training runs
# Uses 2 A100 GPUs with 2 concurrent jobs

# Setup environment
module load cuda/12.3
source activate loki

# Navigate to repo
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

# Create log directories
mkdir -p log/slurm
mkdir -p log/train_loki_task/locomotion
mkdir -p log/train_loki_task/obstacle
mkdir -p log/train_loki_task/incline

# Print job info
echo "=== LOKI Evaluation Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

nvidia-smi
echo ""

# Run the batch launcher: 2 concurrent jobs, 10GB memory threshold, 2 GPUs
bash scripts/train_loki_all_cluster_tasks.sh 2 10000 2

echo ""
echo "End time: $(date)"
echo ""
echo "Check results with:"
echo "  ls metamorph/output/loki/*/kmeans_cluster/20/*/walker20/freq2/drop2/seed3429/Unimal-v0.pt"
