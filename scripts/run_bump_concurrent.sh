#!/bin/bash
#SBATCH --job-name=loki-bump-conc
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-bump-conc-%j.out
#SBATCH --error=log/slurm/loki-bump-conc-%j.err

# === Comparison Experiment: Concurrent mode ===
# 1 H200 GPU, 4 evaluations run concurrently with 5-min stagger
# Clusters: 2, 3, 4, 5 | Task: bump
#
# Submit: sbatch --gres=gpu:h200:1 scripts/run_bump_concurrent.sh

module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

mkdir -p log/slurm log/train_loki_task/bump

STAGGER_TIME=300  # 5 minutes

echo "=== Bump Evaluation: CONCURRENT mode (1 GPU, 4 evals, ${STAGGER_TIME}s stagger) ==="
echo "Clusters: 2, 3, 4, 5"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

CLUSTERS=(2 3 4 5)
PIDS=()

cleanup() {
    echo ""
    echo "[CLEANUP] Received signal, killing all child processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            echo "  Killed PID $pid"
        fi
    done
    wait 2>/dev/null
    echo "[CLEANUP] Done."
    exit 1
}
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# Launch with 5-min stagger
for i in "${!CLUSTERS[@]}"; do
    cluster="${CLUSTERS[$i]}"

    echo "[$(date)] Launching cluster $cluster on GPU 0 (PID will follow)..."
    bash scripts/train_loki_task.sh 20 20 $cluster 3429 bump 0 &
    pid=$!
    PIDS+=($pid)
    echo "[$(date)] Cluster $cluster launched (PID=$pid)"

    # Stagger: wait 5 min before next launch (skip after last one)
    if [ "$i" -lt $(( ${#CLUSTERS[@]} - 1 )) ]; then
        echo "[$(date)] Waiting ${STAGGER_TIME}s before next launch..."
        sleep $STAGGER_TIME
    fi
done

echo ""
echo "[$(date)] All 4 evaluations launched. Waiting for completion..."
echo ""

# Wait for all to finish and report results
for i in "${!CLUSTERS[@]}"; do
    cluster="${CLUSTERS[$i]}"
    pid="${PIDS[$i]}"
    wait "$pid"
    exit_code=$?
    echo "[$(date)] Cluster $cluster (PID=$pid) finished (exit code: $exit_code)"
done

echo ""
echo "=== All concurrent evaluations done ==="
echo "End time: $(date)"

trap - EXIT
