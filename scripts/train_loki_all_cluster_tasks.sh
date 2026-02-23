#!/bin/bash
# Resource-aware batch launcher for LOKI cluster-task training.
# Monitors GPU memory and CPU load before launching new jobs.
# Skips already-completed runs (checks for Unimal-v0.pt).
#
# Usage: bash scripts/train_loki_all_cluster_tasks.sh [max_concurrent_jobs] [gpu_mem_threshold_mb]
#
# Example: bash scripts/train_loki_all_cluster_tasks.sh 2 10000
#   - Runs at most 2 concurrent jobs
#   - Only launches a new job if >= 10000 MiB GPU memory is free

MAX_CONCURRENT=${1:-2}           # Max concurrent jobs (default: 2)
GPU_MEM_THRESHOLD=${2:-10000}    # Min free GPU memory in MiB to launch (default: 10000)
GPU_ID=0                         # GPU device index

NUM_WALKER=20
NUM_CLUSTERS=20
RNG_SEED=3429
TASKS=("locomotion" "obstacle" "incline")

DROP_FREQ=2
NUM_DROP=2

# Map task to env_type for output path matching
get_env_type() {
    case "$1" in
        locomotion) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

# Get free GPU memory in MiB
get_free_gpu_mem() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d ' '
}

# Count currently running training jobs (by checking for train_loki.py processes)
count_running_jobs() {
    pgrep -f "python tools/train_loki.py" | wc -l
}

# Check if a job is already completed
is_completed() {
    local task=$1
    local cluster=$2
    local env_type=$(get_env_type "$task")
    local ckpt="metamorph/output/loki/$env_type/kmeans_cluster/$NUM_CLUSTERS/$cluster/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED/Unimal-v0.pt"
    [ -f "$ckpt" ]
}

# Wait until resources are available
wait_for_resources() {
    while true; do
        local running=$(count_running_jobs)
        local free_mem=$(get_free_gpu_mem)

        if [ "$running" -lt "$MAX_CONCURRENT" ] && [ "$free_mem" -gt "$GPU_MEM_THRESHOLD" ]; then
            return 0
        fi

        echo "  Waiting... (running=$running/$MAX_CONCURRENT, free_gpu=${free_mem}MiB/${GPU_MEM_THRESHOLD}MiB threshold)"
        sleep 60
    done
}

# --- Main ---
echo "=== LOKI Cluster-Task Batch Launcher ==="
echo "Tasks: ${TASKS[*]}"
echo "Clusters: 0-$((NUM_CLUSTERS-1))"
echo "Max concurrent jobs: $MAX_CONCURRENT"
echo "GPU memory threshold: ${GPU_MEM_THRESHOLD} MiB"
echo ""

TOTAL_JOBS=$((NUM_CLUSTERS * ${#TASKS[@]}))
COMPLETED=0
SKIPPED=0
LAUNCHED=0

for TASK in "${TASKS[@]}"; do
    for CLUSTER_LABEL in $(seq 0 $((NUM_CLUSTERS-1))); do
        JOB_ID="task=${TASK} cluster=${CLUSTER_LABEL}"

        # Skip if already completed
        if is_completed "$TASK" "$CLUSTER_LABEL"; then
            echo "[SKIP] $JOB_ID (already completed)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Wait for resources
        echo "[WAIT] $JOB_ID - checking resources..."
        wait_for_resources

        # Launch job
        echo "[LAUNCH] $JOB_ID"
        bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $CLUSTER_LABEL $RNG_SEED $TASK
        LAUNCHED=$((LAUNCHED + 1))

        # Brief pause to let the process start and allocate GPU memory
        sleep 10
    done
done

echo ""
echo "=== Batch launcher done ==="
echo "Total jobs: $TOTAL_JOBS"
echo "Skipped (already done): $SKIPPED"
echo "Launched: $LAUNCHED"
echo ""
echo "Jobs may still be running in background. Monitor with:"
echo "  pgrep -af 'train_loki.py'"
echo "  nvidia-smi"
echo "  tail -f log/train_loki_task/<task>/*.log"
