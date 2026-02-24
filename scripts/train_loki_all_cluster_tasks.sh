#!/bin/bash
# Resource-aware batch launcher for LOKI cluster-task training.
# Monitors GPU memory, system RAM, and CPU load before launching new jobs.
# Skips already-completed runs (checks for Unimal-v0.pt).
# Distributes jobs across multiple GPUs in round-robin fashion.
# Properly tracks and cleans up child processes on exit.
#
# Usage: bash scripts/train_loki_all_cluster_tasks.sh [max_concurrent_jobs] [gpu_mem_threshold_mb] [num_gpus] [ram_threshold_mb]
#
# Example: bash scripts/train_loki_all_cluster_tasks.sh 2 10000 2 60000
#   - Runs at most 2 concurrent jobs
#   - Only launches a new job if >= 10000 MiB GPU memory is free
#   - Distributes across 2 GPUs
#   - Only launches a new job if >= 60000 MiB system RAM is free

MAX_CONCURRENT=${1:-2}           # Max concurrent jobs (default: 2)
GPU_MEM_THRESHOLD=${2:-10000}    # Min free GPU memory in MiB to launch (default: 10000)
NUM_GPUS=${3:-2}                 # Number of GPUs available (default: 2)
RAM_THRESHOLD=${4:-60000}        # Min free system RAM in MiB to launch (default: 60000)
NEXT_GPU=0                       # Round-robin GPU assignment counter

NUM_WALKER=20
NUM_CLUSTERS=20
RNG_SEED=3429
TASKS=("locomotion" "obstacle" "incline")

DROP_FREQ=2
NUM_DROP=2

# Track child PIDs and their job descriptions for status display
declare -A PID_TO_JOB     # Maps PID -> job description string
CHILD_PIDS=()

# Cleanup handler: kill all child processes on exit/signal
cleanup() {
    echo ""
    echo "[CLEANUP] Received signal, killing all child processes..."
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            echo "  Killed PID $pid (${PID_TO_JOB[$pid]:-unknown})"
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

# Get available system RAM in MiB
get_free_ram() {
    free -m 2>/dev/null | awk '/^Mem:/ {print $7}'
}

# Count currently running training jobs and update PID list
# Also prints status of running/waiting jobs when called with "verbose" argument
count_running_jobs() {
    local count=0
    local alive_pids=()
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
            alive_pids+=("$pid")
        else
            # Job finished, clean up its entry
            unset PID_TO_JOB[$pid]
        fi
    done
    CHILD_PIDS=("${alive_pids[@]}")
    echo $count
}

# Display status of all running and pending jobs
print_job_status() {
    local pending_job="$1"
    echo "  ---- Job Status ----"
    # Show running jobs
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  [RUNNING]  ${PID_TO_JOB[$pid]}  (PID=$pid)"
        fi
    done
    # Show the pending job that's waiting
    if [ -n "$pending_job" ]; then
        echo "  [WAITING]  $pending_job"
    fi
    echo "  ---------------------"
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
    local pending_job="$1"
    local first_wait=true
    while true; do
        local running=$(count_running_jobs)
        local free_gpu=$(get_free_gpu_mem)
        local free_ram=$(get_free_ram)

        if [ "$running" -lt "$MAX_CONCURRENT" ] && \
           [ "$free_gpu" -gt "$GPU_MEM_THRESHOLD" ] && \
           [ "$free_ram" -gt "$RAM_THRESHOLD" ]; then
            return 0
        fi

        if [ "$first_wait" = true ]; then
            print_job_status "$pending_job"
            first_wait=false
        fi

        echo "  Waiting... (running=$running/$MAX_CONCURRENT, free_gpu=${free_gpu}MiB/${GPU_MEM_THRESHOLD}MiB, free_ram=${free_ram}MiB/${RAM_THRESHOLD}MiB)"
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
echo "RAM threshold: ${RAM_THRESHOLD} MiB"
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

        # Wait for resources (pass pending job name for status display)
        echo "[WAIT] $JOB_ID - checking resources..."
        wait_for_resources "$JOB_ID"

        # Launch job on next GPU (round-robin)
        # The & here backgrounds the subshell so $! correctly captures its PID
        bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $CLUSTER_LABEL $RNG_SEED $TASK $NEXT_GPU &
        local_pid=$!
        CHILD_PIDS+=($local_pid)
        PID_TO_JOB[$local_pid]="$JOB_ID (GPU=$NEXT_GPU)"

        echo "[LAUNCH] $JOB_ID on GPU $NEXT_GPU (PID=$local_pid)"
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
