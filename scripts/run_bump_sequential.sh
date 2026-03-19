#!/bin/bash
#SBATCH --job-name=loki-bump-seq
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-bump-seq-%j.out
#SBATCH --error=log/slurm/loki-bump-seq-%j.err

# === Comparison Experiment: Sequential mode ===
# 1 H200 GPU, 2 evaluations run sequentially (one after the other)
# Clusters: 0, 1 | Task: bump
#
# Submit: sbatch --gres=gpu:h200:1 scripts/run_bump_sequential.sh

module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

mkdir -p log/slurm log/train_loki_task/bump

echo "=== Bump Evaluation: SEQUENTIAL mode (1 GPU, 2 evals) ==="
echo "Clusters: 0, 1"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Sequential: cluster 0 runs to completion, then cluster 1
echo "[$(date)] Starting cluster 0..."
bash scripts/train_loki_task.sh 20 20 0 3429 bump 0
echo "[$(date)] Cluster 0 finished (exit code: $?)"

echo ""
echo "[$(date)] Starting cluster 1..."
bash scripts/train_loki_task.sh 20 20 1 3429 bump 0
echo "[$(date)] Cluster 1 finished (exit code: $?)"

echo ""
echo "=== All sequential evaluations done ==="
echo "End time: $(date)"
