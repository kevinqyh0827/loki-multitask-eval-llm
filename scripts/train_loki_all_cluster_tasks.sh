#!/bin/bash
# Profiling-based adaptive batch launcher for LOKI cluster-task training.
#
# Instead of fixed thresholds, this script:
#   1. Launches an initial batch of jobs (one per GPU)
#   2. Waits for them to stabilize (~5 min) and measures per-job resource usage
#   3. Auto-calculates how many concurrent jobs the node can handle
#   4. Schedules remaining jobs with proper stabilization gaps
#
# Skips already-completed runs (checks for Unimal-v0.pt).
# Distributes jobs across GPUs in round-robin fashion.
# Displays running/queued job status during waits.
#
# Usage: bash scripts/train_loki_all_cluster_tasks.sh [num_gpus] [profiling_stabilize_s] [phase2_stabilize_s]
#
# Example: bash scripts/train_loki_all_cluster_tasks.sh 2 300 30
#   - Uses 2 GPUs, 300s profiling stabilization, 30s between Phase 2 launches

NUM_GPUS=${1:-2}                 # Number of GPUs available (default: 2)
STABILIZE_TIME=${2:-300}         # Seconds to wait after launch for resource stabilization (default: 5 min)
PHASE2_STABILIZE=${3:-30}        # Seconds between launches in Phase 2 (default: 30s)
SAFETY_MARGIN=90                 # Use only 90% of measured capacity (reserve 10% for eval bursts)

NUM_WALKER=20
NUM_CLUSTERS=20
RNG_SEED=3429
TASKS=("locomotion" "obstacle" "incline")

DROP_FREQ=2
NUM_DROP=2

# Scheduling state (set during profiling phase)
GPU_PER_JOB=0
RAM_PER_JOB=0
MAX_CONCURRENT=0
NEXT_GPU=0

# Track child PIDs, job descriptions, and start times
declare -A PID_TO_JOB       # PID -> "task=X cluster=Y (GPU=Z)"
declare -A PID_TO_START     # PID -> epoch seconds at launch
CHILD_PIDS=()

# Job queue: arrays of task/cluster pairs to run
QUEUE_TASKS=()
QUEUE_CLUSTERS=()
QUEUE_IDX=0

# Counters
TOTAL_JOBS=0
SKIPPED=0
COMPLETED=0
LAUNCHED=0

# --- Utility functions ---

cleanup() {
    echo ""
    echo "[CLEANUP] Received signal, killing all child processes..."
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            echo "  Killed PID $pid (${PID_TO_JOB[$pid]:-unknown})"
        fi
    done
    pkill -P $$ 2>/dev/null
    wait 2>/dev/null
    echo "[CLEANUP] Done."
    exit 1
}

trap cleanup SIGTERM SIGINT SIGHUP EXIT

get_env_type() {
    case "$1" in
        locomotion) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

is_completed() {
    local task=$1
    local cluster=$2
    local env_type=$(get_env_type "$task")
    local ckpt="metamorph/output/loki/$env_type/kmeans_cluster/$NUM_CLUSTERS/$cluster/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED/Unimal-v0.pt"
    [ -f "$ckpt" ]
}

# Get total free GPU memory (sum across all GPUs) in MiB
get_free_gpu_mem() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}'
}

# Get total GPU memory (sum across all GPUs) in MiB
get_total_gpu_mem() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}'
}

# Get free memory for a specific GPU index, in MiB
get_gpu_free_mem() {
    local gpu_idx=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$gpu_idx 2>/dev/null | tr -d ' '
}

# Pick the GPU with the most free memory; sets NEXT_GPU and prints its free MiB
pick_best_gpu() {
    local best_gpu=0
    local best_free=0
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        local free=$(get_gpu_free_mem $gpu)
        if [ "$free" -gt "$best_free" ]; then
            best_free=$free
            best_gpu=$gpu
        fi
    done
    NEXT_GPU=$best_gpu
    echo $best_free
}

# Get available system RAM in MiB
get_free_ram() {
    free -m 2>/dev/null | awk '/^Mem:/ {print $7}'
}

# Get total system RAM in MiB
get_total_ram() {
    free -m 2>/dev/null | awk '/^Mem:/ {print $2}'
}

# Count running jobs and prune dead PIDs
count_running_jobs() {
    local count=0
    local alive_pids=()
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
            alive_pids+=("$pid")
        else
            unset PID_TO_JOB[$pid]
            unset PID_TO_START[$pid]
            COMPLETED=$((COMPLETED + 1))
        fi
    done
    CHILD_PIDS=("${alive_pids[@]}")
    echo $count
}

# Format elapsed time as "Xh Ym"
format_elapsed() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    if [ "$hours" -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

# Display comprehensive job status
print_status() {
    local running=$(count_running_jobs)
    local queued=$(( ${#QUEUE_TASKS[@]} - QUEUE_IDX ))
    local now=$(date +%s)

    echo ""
    echo "=== Job Status (${running} running / ${queued} queued / ${COMPLETED} completed / ${SKIPPED} skipped) ==="

    # Show running jobs
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            local elapsed=$(( now - ${PID_TO_START[$pid]} ))
            echo "  [RUNNING]  ${PID_TO_JOB[$pid]}  PID=$pid  ($(format_elapsed $elapsed))"
        fi
    done

    # Show next queued job
    if [ "$QUEUE_IDX" -lt "${#QUEUE_TASKS[@]}" ]; then
        echo "  [QUEUED]   task=${QUEUE_TASKS[$QUEUE_IDX]} cluster=${QUEUE_CLUSTERS[$QUEUE_IDX]}  (next)"
    fi

    # Show resource usage
    local free_gpu=$(get_free_gpu_mem)
    local total_gpu=$(get_total_gpu_mem)
    local free_ram=$(get_free_ram)
    local total_ram=$(get_total_ram)
    local used_gpu=$((total_gpu - free_gpu))
    local used_ram=$((total_ram - free_ram))

    echo "  Resources: GPU ${used_gpu}/${total_gpu} MiB used | RAM ${used_ram}/${total_ram} MiB used"
    if [ "$GPU_PER_JOB" -gt 0 ]; then
        echo "  Per-job: ~${GPU_PER_JOB} MiB GPU, ~${RAM_PER_JOB} MiB RAM | Max concurrent: $MAX_CONCURRENT"
    fi

    # Per-GPU breakdown
    echo "  --- Per-GPU Usage ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
        echo "    GPU $line"
    done
    echo "================================================================"
    echo ""
}

# Launch the next job from the queue
launch_next_job() {
    if [ "$QUEUE_IDX" -ge "${#QUEUE_TASKS[@]}" ]; then
        return 1  # No more jobs
    fi

    local task="${QUEUE_TASKS[$QUEUE_IDX]}"
    local cluster="${QUEUE_CLUSTERS[$QUEUE_IDX]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))

    local job_desc="task=${task} cluster=${cluster} (GPU=$NEXT_GPU)"

    bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $cluster $RNG_SEED $task $NEXT_GPU &
    local pid=$!
    CHILD_PIDS+=($pid)
    PID_TO_JOB[$pid]="$job_desc"
    PID_TO_START[$pid]=$(date +%s)

    echo "[LAUNCH] $job_desc (PID=$pid)"
    # Pick GPU with most free memory for next launch (replaces round-robin)
    pick_best_gpu > /dev/null
    LAUNCHED=$((LAUNCHED + 1))

    return 0
}

# Measure per-job resource consumption after stabilization (called once during profiling)
measure_per_job_usage() {
    local running=$(count_running_jobs)
    if [ "$running" -eq 0 ]; then
        echo "[ERROR] No running jobs to measure. Aborting."
        exit 1
    fi

    local total_gpu=$(get_total_gpu_mem)
    local free_gpu=$(get_free_gpu_mem)
    local total_ram=$(get_total_ram)
    local free_ram=$(get_free_ram)

    local used_gpu=$((total_gpu - free_gpu))
    local used_ram=$((total_ram - free_ram))

    GPU_PER_JOB=$((used_gpu / running))
    RAM_PER_JOB=$((used_ram / running))
}

# Dynamically calculate how many concurrent jobs can fit based on current resources.
# Called every scheduling iteration so MAX_CONCURRENT adapts as jobs start/finish.
calculate_max_concurrent() {
    local total_gpu=$(get_total_gpu_mem)
    local total_ram=$(get_total_ram)

    # Apply safety margin: only use SAFETY_MARGIN% of total resources
    local safe_gpu=$(( total_gpu * SAFETY_MARGIN / 100 ))
    local safe_ram=$(( total_ram * SAFETY_MARGIN / 100 ))

    local max_by_gpu=$((safe_gpu / GPU_PER_JOB))
    local max_by_ram=$((safe_ram / RAM_PER_JOB))

    # Take the minimum of GPU and RAM limits, but at least 1
    if [ "$max_by_gpu" -lt "$max_by_ram" ]; then
        MAX_CONCURRENT=$max_by_gpu
    else
        MAX_CONCURRENT=$max_by_ram
    fi
    if [ "$MAX_CONCURRENT" -lt 1 ]; then
        MAX_CONCURRENT=1
    fi
}

# --- Build job queue ---

echo "=== LOKI Cluster-Task Adaptive Batch Launcher ==="
echo "Num GPUs: $NUM_GPUS"
echo "Profiling stabilization: ${STABILIZE_TIME}s"
echo "Phase 2 stabilization: ${PHASE2_STABILIZE}s"
echo "Safety margin: ${SAFETY_MARGIN}%"
echo ""

# Build queue of pending jobs (skip completed ones)
for TASK in "${TASKS[@]}"; do
    for CLUSTER_LABEL in $(seq 0 $((NUM_CLUSTERS-1))); do
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        if is_completed "$TASK" "$CLUSTER_LABEL"; then
            echo "[SKIP] task=${TASK} cluster=${CLUSTER_LABEL} (already completed)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        QUEUE_TASKS+=("$TASK")
        QUEUE_CLUSTERS+=("$CLUSTER_LABEL")
    done
done

PENDING=${#QUEUE_TASKS[@]}
echo ""
echo "Total: $TOTAL_JOBS | Already completed: $SKIPPED | To run: $PENDING"

if [ "$PENDING" -eq 0 ]; then
    echo "All jobs already completed!"
    trap - EXIT
    exit 0
fi

# --- Phase 1: Profiling ---

echo ""
echo "=== Phase 1: Profiling (launching initial batch) ==="

# Launch one job per GPU as initial batch
INITIAL_BATCH=$NUM_GPUS
if [ "$INITIAL_BATCH" -gt "$PENDING" ]; then
    INITIAL_BATCH=$PENDING
fi

for i in $(seq 1 $INITIAL_BATCH); do
    launch_next_job
done

echo ""
echo "[PROFILING] Waiting ${STABILIZE_TIME}s for jobs to fully initialize and allocate resources..."
echo "[PROFILING] (This ensures accurate resource measurement)"

# Wait for stabilization, showing countdown every 60s
elapsed=0
while [ "$elapsed" -lt "$STABILIZE_TIME" ]; do
    sleep_chunk=60
    remaining=$((STABILIZE_TIME - elapsed))
    if [ "$sleep_chunk" -gt "$remaining" ]; then
        sleep_chunk=$remaining
    fi
    sleep $sleep_chunk
    elapsed=$((elapsed + sleep_chunk))

    # Check if initial jobs are still alive
    local_running=$(count_running_jobs)
    if [ "$local_running" -eq 0 ]; then
        echo "[WARNING] All initial jobs exited during profiling. Check logs for errors."
        echo "Attempting to continue with remaining jobs..."
        break
    fi
    echo "  Stabilizing... ${elapsed}/${STABILIZE_TIME}s (${local_running} jobs running)"
done

# Measure resource usage
local_running=$(count_running_jobs)
if [ "$local_running" -gt 0 ]; then
    measure_per_job_usage
    calculate_max_concurrent
    echo ""
    echo "[PROFILING] Measurement complete:"
    echo "  Running jobs: $local_running"
    echo "  GPU per job: ~${GPU_PER_JOB} MiB"
    echo "  RAM per job: ~${RAM_PER_JOB} MiB"
    echo "  Max concurrent (with ${SAFETY_MARGIN}% safety margin): $MAX_CONCURRENT"
else
    echo "[WARNING] No jobs running after profiling. Setting MAX_CONCURRENT=1"
    MAX_CONCURRENT=1
fi

# --- Phase 2: Steady-state scheduling ---

echo ""
echo "=== Phase 2: Steady-state scheduling ==="

while true; do
    running=$(count_running_jobs)
    pending_left=$(( ${#QUEUE_TASKS[@]} - QUEUE_IDX ))

    # Exit when no running jobs and no pending jobs
    if [ "$running" -eq 0 ] && [ "$pending_left" -eq 0 ]; then
        break
    fi

    # Recalculate max concurrent based on current resources
    if [ "$GPU_PER_JOB" -gt 0 ]; then
        old_max=$MAX_CONCURRENT
        calculate_max_concurrent
        if [ "$MAX_CONCURRENT" -ne "$old_max" ]; then
            echo "[SCHEDULER] Max concurrent updated: $old_max -> $MAX_CONCURRENT"
        fi
    fi

    # Try to launch ONE job per cycle, then wait for it to allocate resources
    if [ "$pending_left" -gt 0 ] && [ "$running" -lt "$MAX_CONCURRENT" ]; then
        # Check per-GPU free memory (not total across all GPUs)
        best_free=$(pick_best_gpu)
        free_ram=$(get_free_ram)

        if [ "$best_free" -gt "$GPU_PER_JOB" ] && [ "$free_ram" -gt "$RAM_PER_JOB" ]; then
            launch_next_job
            print_status
            echo "[STABILIZE] Waiting ${PHASE2_STABILIZE}s for resource allocation..."
            sleep $PHASE2_STABILIZE
            continue
        fi
    fi

    # Nothing to launch right now, show status and wait
    print_status
    sleep 60
done

# --- Done ---

echo ""
echo "=== All jobs completed ==="
echo "Total: $TOTAL_JOBS | Completed: $COMPLETED | Skipped: $SKIPPED | Launched: $LAUNCHED"

trap - EXIT
