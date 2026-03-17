#!/bin/bash
#SBATCH --job-name=loki-ft-012
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-ft-012-%j.out
#SBATCH --error=log/slurm/loki-ft-012-%j.err

# === Retrain locomotion clusters 0, 1, 2 (fresh runs) ===
# These 3 runs were cleaned and need to be re-evaluated.
#
# Submit: sbatch --gres=gpu:h200:1 scripts/run_ft_012.sh

# Setup environment
module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki

# Navigate to repo
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

# Create log directories
mkdir -p log/slurm
mkdir -p log/train_loki_task/locomotion

# Print job info
echo "=== Retraining locomotion clusters 0, 1, 2 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Launch all 3 on GPU 0 with 5-min stagger
# Args: num_walkers num_clusters cluster_idx seed task gpu_id
bash scripts/train_loki_task.sh 20 20 0 3429 locomotion 0 &
PID0=$!
echo "[LAUNCH] cluster=0 PID=$PID0"

sleep 300

bash scripts/train_loki_task.sh 20 20 1 3429 locomotion 0 &
PID1=$!
echo "[LAUNCH] cluster=1 PID=$PID1"

sleep 300

bash scripts/train_loki_task.sh 20 20 2 3429 locomotion 0 &
PID2=$!
echo "[LAUNCH] cluster=2 PID=$PID2"

# Wait for all to finish
echo ""
echo "All 3 jobs launched. Waiting for completion..."
wait $PID0
echo "[DONE] cluster=0 exit=$?"
wait $PID1
echo "[DONE] cluster=1 exit=$?"
wait $PID2
echo "[DONE] cluster=2 exit=$?"

echo ""
echo "All done: $(date)"
