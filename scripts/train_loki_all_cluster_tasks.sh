#!/bin/bash
# Resource-aware batch launcher for LOKI cluster-task training.
# Monitors GPU memory and CPU load before launching new jobs.
# Skips already-completed runs (checks for Unimal-v0.pt).
# Distributes jobs across multiple GPUs in round-robin fashion.
# Properly tracks and cleans up child processes on exit.
#
# Usage: bash scripts/train_loki_all_cluster_tasks.sh [max_concurrent_jobs] [gpu_mem_threshold_mb] [num_gpus]
#
# Example: bash scripts/train_loki_all_cluster_tasks.sh 2 10000 2
#   - Runs at most 2 concurrent jobs
#   - Only launches a new job if >= 10000 MiB GPU memory is free
#   - Distributes across 2 GPUs

MAX_CONCURRENT=${1:-2}           # Max concurrent jobs (default: 2)
GPU_MEM_THRESHOLD=${2:-10000}    # Min free GPU memory in MiB to launch (default: 10000)
NUM_GPUS=${3:-2}                 # Number of GPUs available (default: 2)
NEXT_GPU=0                       # Round-robin GPU assignment counter

NUM_WALKER=20
NUM_CLUSTERS=20
RNG_SEED=3429
TASKS=("locomotion" "obstacle" "incline")

DROP_FREQ=2
NUM_DROP=2

# Track child PIDs for cleanup
CHILD_PIDS=()

# Cleanup handler: kill all child processes on exit/signal
cleanup() {
    echo ""
    echo "[CLEANUP] Received signal, killing all child processes..."
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            echo "  Killed PID $pid"
        fi
    done
    # Also kill any remaining train_loki.py processes in our process group
    pkill -P $$ 2>/dev/null
    wait 2>/dev/null
    echo "[CLEANUP] Done."
    exit 1
}

trap cleanup SIGTERM SIGINT SIGHUP EXIT

# Map task to env_type for output path matching
get_env_type() {
    case "$1" in
        locomotion) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

# Get minimum free GPU memory across all GPUs in MiB
get_free_gpu_mem() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1 | tr -d ' '
}

# Count currently running training jobs from our tracked PIDs
count_running_jobs() {
    local count=0
    local alive_pids=()
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
            alive_pids+=("$pid")
        fi
    done
    CHILD_PIDS=("${alive_pids[@]}")
    echo $count
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
echo "Num GPUs: $NUM_GPUS"
echo "GPU memory threshold: ${GPU_MEM_THRESHOLD} MiB"
echo ""

TOTAL_JOBS=$((NUM_CLUSTERS * ${#TASKS[@]}))
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

        # Launch job on next GPU (round-robin)
        echo "[LAUNCH] $JOB_ID on GPU $NEXT_GPU"
        bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $CLUSTER_LABEL $RNG_SEED $TASK $NEXT_GPU
        CHILD_PIDS+=($!)
        NEXT_GPU=$(( (NEXT_GPU + 1) % NUM_GPUS ))
        LAUNCHED=$((LAUNCHED + 1))

        # Brief pause to let the process start and allocate GPU memory
        sleep 10
    done
done

echo ""
echo "=== Batch launcher: all jobs submitted ==="
echo "Total jobs: $TOTAL_JOBS"
echo "Skipped (already done): $SKIPPED"
echo "Launched: $LAUNCHED"
echo ""
echo "Waiting for remaining jobs to finish..."

# Wait for all tracked children
for pid in "${CHILD_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
    fi
done

# Disable cleanup trap on normal exit
trap - EXIT
echo "=== All jobs completed ==="
