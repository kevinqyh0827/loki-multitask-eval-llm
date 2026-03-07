#!/bin/bash
#SBATCH --job-name=loki-eval
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-eval-%j.out
#SBATCH --error=log/slurm/loki-eval-%j.err

# === LOKI Cluster-Task Evaluation on Palmetto ===
# Runs 3 tasks (locomotion, obstacle, incline) x 20 clusters = 60 training runs
# Auto-detects number of GPUs on the node.
#
# Submit examples:
#   sbatch --gres=gpu:a100:2 scripts/run_eval_cluster.sh   # 2x A100
#   sbatch --gres=gpu:h200:4 scripts/run_eval_cluster.sh   # 4x H200
#   sbatch --gres=gpu:8 scripts/run_eval_cluster.sh        # 8x any GPU

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
mkdir -p log/train_loki_task/incline

# Print job and node info
echo "=== LOKI Evaluation Job ==="
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

# Auto-detect number of GPUs on this node
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo ""

# Run the sequential scheduler with detected GPU count
# Args: num_gpus, max_concurrent_per_gpu (0=auto-detect), test_mode (0=full, 1=reduced steps)
# Override via environment: TEST_MODE=1 sbatch ... to run with reduced steps (1e5)
bash scripts/train_loki_all_cluster_tasks.sh $NUM_GPUS 0 ${TEST_MODE:-0}

echo ""
echo "End time: $(date)"
echo ""
echo "Check results with:"
echo "  ls metamorph/output/loki/*/kmeans_cluster/20/*/walker20/freq2/drop2/seed3429/Unimal-v0_results.json"
