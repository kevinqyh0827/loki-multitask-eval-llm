#!/bin/bash
# Single-task, single-GPU sequential scheduler for LOKI cluster evaluation.
#
# Runs all 20 clusters for ONE task on ONE GPU, launching evaluations
# one at a time with a 5-minute stagger between each launch.
# Max concurrent evaluations is set by GPU memory tier (or override).
# Pauses scheduling when system RAM drops below threshold.
#
# Usage: bash scripts/train_loki_single_task.sh <task_name> [max_concurrent] [test_mode]
#
# Examples:
#   bash scripts/train_loki_single_task.sh locomotion         # auto-detect concurrency
#   bash scripts/train_loki_single_task.sh obstacle 10        # force max 10 concurrent
#   bash scripts/train_loki_single_task.sh incline 0 1        # auto-detect, test mode (1e5 steps)

TASK_NAME=${1:?  "Usage: $0 <task_name> [max_concurrent] [test_mode]"}
MAX_CONCURRENT_OVERRIDE=${2:-0}      # 0 = auto-detect from GPU memory tier
TEST_MODE=${3:-0}                    # 1 = reduced steps for testing (1e5 instead of 1e8)

STAGGER_TIME=300                     # 5 minutes between launches
RAM_THRESHOLD=80000                  # Minimum free system RAM in MiB (80 GB)
GPU_ID=0                             # Always use GPU 0 (SLURM remaps to logical index 0)

NUM_WALKER=20
NUM_CLUSTERS=20
RNG_SEED=3429
DROP_FREQ=2
NUM_DROP=2

# Scheduling state
MAX_CONCURRENT=0
FAILURE_COOLDOWN_UNTIL=0

# PID tracking
declare -A PID_TO_JOB
declare -A PID_TO_START
CHILD_PIDS=()

# Job queue
QUEUE_CLUSTERS=()
QUEUE_IDX=0

# Counters
TOTAL_JOBS=0
SKIPPED=0
COMPLETED=0
LAUNCHED=0
RUNNING_COUNT=0

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
    local cluster=$1
    local env_type=$(get_env_type "$TASK_NAME")
    local ckpt="metamorph/output/loki/$env_type/kmeans_cluster/$NUM_CLUSTERS/$cluster/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED/Unimal-v0.pt"
    [ -f "$ckpt" ]
}

get_gpu_total_mem() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$GPU_ID 2>/dev/null | tr -d ' '
}

get_gpu_free_mem() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$GPU_ID 2>/dev/null | tr -d ' '
}

get_free_ram() {
    free -m 2>/dev/null | awk '/^Mem:/ {print $7}'
}

get_total_ram() {
    free -m 2>/dev/null | awk '/^Mem:/ {print $2}'
}

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

detect_max_concurrent() {
    local gpu_mem=$(get_gpu_total_mem)
    if [ "$gpu_mem" -le 50000 ]; then
        MAX_CONCURRENT=5
    elif [ "$gpu_mem" -le 100000 ]; then
        MAX_CONCURRENT=10
    else
        MAX_CONCURRENT=16
    fi
    echo "[CONFIG] GPU has ${gpu_mem} MiB VRAM -> max ${MAX_CONCURRENT} concurrent"
}

update_running_jobs() {
    RUNNING_COUNT=0
    local alive_pids=()
    local now=$(date +%s)
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            RUNNING_COUNT=$((RUNNING_COUNT + 1))
            alive_pids+=("$pid")
        else
            wait "$pid" 2>/dev/null
            local exit_code=$?
            local job="${PID_TO_JOB[$pid]}"
            local start="${PID_TO_START[$pid]}"
            local runtime=0
            if [ -n "$start" ]; then
                runtime=$(( now - start ))
            fi

            if [ "$exit_code" -eq 0 ]; then
                echo "[DONE] $job completed successfully (runtime: $(format_elapsed $runtime))"
            elif [ "$exit_code" -eq 137 ]; then
                echo "[OOM-KILLED] $job killed after $(format_elapsed $runtime) (exit $exit_code, likely OOM killer)"
                FAILURE_COOLDOWN_UNTIL=$(( now + 60 ))
            elif [ "$runtime" -lt 30 ]; then
                echo "[FAST-FAIL] $job crashed in ${runtime}s (exit $exit_code)"
                FAILURE_COOLDOWN_UNTIL=$(( now + 30 ))
            else
                echo "[FAIL] $job crashed after $(format_elapsed $runtime) (exit $exit_code)"
                FAILURE_COOLDOWN_UNTIL=$(( now + 45 ))
            fi

            COMPLETED=$((COMPLETED + 1))
            unset PID_TO_JOB[$pid]
            unset PID_TO_START[$pid]
        fi
    done
    CHILD_PIDS=("${alive_pids[@]}")
}

launch_next_job() {
    if [ "$QUEUE_IDX" -ge "${#QUEUE_CLUSTERS[@]}" ]; then
        return 1
    fi

    local cluster="${QUEUE_CLUSTERS[$QUEUE_IDX]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))

    local job_desc="task=${TASK_NAME} cluster=${cluster} (GPU=${GPU_ID})"

    if [ "$TEST_MODE" -eq 1 ]; then
        bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $cluster $RNG_SEED $TASK_NAME $GPU_ID 1 &
    else
        bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $cluster $RNG_SEED $TASK_NAME $GPU_ID &
    fi
    local pid=$!
    CHILD_PIDS+=($pid)
    PID_TO_JOB[$pid]="$job_desc"
    PID_TO_START[$pid]=$(date +%s)

    echo "[LAUNCH] $job_desc (PID=$pid) [${RUNNING_COUNT}+1 running]"
    LAUNCHED=$((LAUNCHED + 1))

    return 0
}

print_status() {
    update_running_jobs
    local queued=$(( ${#QUEUE_CLUSTERS[@]} - QUEUE_IDX ))
    local now=$(date +%s)

    echo ""
    echo "=== [${TASK_NAME}] Status (${RUNNING_COUNT}/${MAX_CONCURRENT} running | ${queued} queued | ${COMPLETED} completed | ${SKIPPED} skipped) ==="

    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            local elapsed=$(( now - ${PID_TO_START[$pid]} ))
            echo "  [RUNNING]  ${PID_TO_JOB[$pid]}  PID=$pid  ($(format_elapsed $elapsed))"
        fi
    done

    if [ "$QUEUE_IDX" -lt "${#QUEUE_CLUSTERS[@]}" ]; then
        echo "  [QUEUED]   cluster=${QUEUE_CLUSTERS[$QUEUE_IDX]}  (next)"
    fi

    local gpu_free=$(get_gpu_free_mem)
    local gpu_total=$(get_gpu_total_mem)
    local free_ram=$(get_free_ram)
    local total_ram=$(get_total_ram)

    echo "  Resources: GPU ${gpu_free}/${gpu_total} MiB free | RAM ${free_ram}/${total_ram} MiB free (threshold: ${RAM_THRESHOLD} MiB)"

    local gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader --id=$GPU_ID 2>/dev/null)
    echo "  GPU $GPU_ID: ${gpu_info}"

    if [ "$now" -lt "$FAILURE_COOLDOWN_UNTIL" ]; then
        local cd_remaining=$(( FAILURE_COOLDOWN_UNTIL - now ))
        echo "  [COOLDOWN] ${cd_remaining}s remaining after failure"
    fi
    echo "================================================================"
    echo ""
}

# Wait with countdown, updating running jobs each minute
wait_stagger() {
    local label=$1
    echo "[${label}] Waiting ${STAGGER_TIME}s..."
    local elapsed=0
    while [ "$elapsed" -lt "$STAGGER_TIME" ]; do
        local sleep_chunk=60
        local remaining=$((STAGGER_TIME - elapsed))
        if [ "$sleep_chunk" -gt "$remaining" ]; then
            sleep_chunk=$remaining
        fi
        sleep $sleep_chunk
        elapsed=$((elapsed + sleep_chunk))
        update_running_jobs
        echo "  ${label}: ${elapsed}/${STAGGER_TIME}s (${RUNNING_COUNT} jobs running)"
    done
}

# --- Main ---

echo "=== LOKI Single-Task Sequential Scheduler ==="
echo "Task: $TASK_NAME"
echo "GPU: $GPU_ID"
echo "Clusters: 0-$((NUM_CLUSTERS-1))"
echo "Max concurrent override: ${MAX_CONCURRENT_OVERRIDE} (0 = auto-detect)"
echo "Stagger time: ${STAGGER_TIME}s (5 min)"
echo "RAM threshold: ${RAM_THRESHOLD} MiB"
echo "Test mode: ${TEST_MODE}"
echo ""

# Detect or set max concurrent
if [ "$MAX_CONCURRENT_OVERRIDE" -gt 0 ]; then
    MAX_CONCURRENT=$MAX_CONCURRENT_OVERRIDE
    echo "[CONFIG] Using override: ${MAX_CONCURRENT} concurrent"
else
    detect_max_concurrent
fi
echo ""

# Build queue (skip completed clusters)
for CLUSTER_LABEL in $(seq 0 $((NUM_CLUSTERS-1))); do
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
    if is_completed "$CLUSTER_LABEL"; then
        echo "[SKIP] cluster=${CLUSTER_LABEL} (already completed)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    QUEUE_CLUSTERS+=("$CLUSTER_LABEL")
done

PENDING=${#QUEUE_CLUSTERS[@]}
echo ""
echo "Total: $TOTAL_JOBS | Already completed: $SKIPPED | To run: $PENDING"

if [ "$PENDING" -eq 0 ]; then
    echo "All clusters already completed for task=${TASK_NAME}!"
    trap - EXIT
    exit 0
fi

# --- Phase 1: Launch first evaluation and measure ---

echo ""
echo "=== Phase 1: Launch first evaluation and measure resources ==="

launch_next_job
wait_stagger "PHASE 1"
print_status

# --- Phase 2: Sequential scheduling ---

echo ""
echo "=== Phase 2: Sequential scheduling (max ${MAX_CONCURRENT} concurrent, ${STAGGER_TIME}s stagger) ==="

while true; do
    update_running_jobs
    pending_left=$(( ${#QUEUE_CLUSTERS[@]} - QUEUE_IDX ))

    # Exit when no running jobs and no pending jobs
    if [ "$RUNNING_COUNT" -eq 0 ] && [ "$pending_left" -eq 0 ]; then
        break
    fi

    # Try to launch if there are pending jobs
    if [ "$pending_left" -gt 0 ]; then
        now=$(date +%s)

        # Check failure cooldown
        if [ "$now" -lt "$FAILURE_COOLDOWN_UNTIL" ]; then
            remaining_cooldown=$(( FAILURE_COOLDOWN_UNTIL - now ))
            echo "[COOLDOWN] Failure cooldown active, ${remaining_cooldown}s remaining..."
            print_status
            sleep 60
            continue
        fi

        # Check max concurrent limit
        if [ "$RUNNING_COUNT" -ge "$MAX_CONCURRENT" ]; then
            echo "[WAIT] At max concurrent limit (${RUNNING_COUNT}/${MAX_CONCURRENT}). Waiting..."
            print_status
            sleep 60
            continue
        fi

        # Check system RAM threshold
        free_ram=$(get_free_ram)
        if [ "$free_ram" -lt "$RAM_THRESHOLD" ]; then
            echo "[WAIT] System RAM too low: ${free_ram} MiB free < ${RAM_THRESHOLD} MiB threshold. Waiting..."
            print_status
            sleep 60
            continue
        fi

        # All conditions met — launch next job
        launch_next_job
        print_status
        wait_stagger "STAGGER"
        continue
    fi

    # No pending jobs, just wait for running jobs to finish
    print_status
    sleep 60
done

# --- Done ---

echo ""
echo "=== Task ${TASK_NAME}: All clusters completed ==="
echo "Total: $TOTAL_JOBS | Completed: $COMPLETED | Skipped: $SKIPPED | Launched: $LAUNCHED"

trap - EXIT
