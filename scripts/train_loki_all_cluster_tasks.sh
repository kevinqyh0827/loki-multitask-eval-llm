#!/bin/bash
# Simplified sequential scheduler for LOKI cluster-task training.
#
# Strategy:
#   1. Builds a task queue of all (task, cluster) combinations
#   2. Phase 1: Launches one job per GPU with 5-minute stagger between each
#   3. Phase 2: Continues launching jobs sequentially (5-min stagger) with
#      fixed max-concurrent limit and system RAM gating (>80 GB free)
#
# Max concurrent evaluations per GPU is determined by GPU memory tier:
#   ~40 GB (A100-40GB):  5 per GPU
#   ~80 GB (A100-80GB): 10 per GPU
#   ~141 GB (H200):     16 per GPU
# Can be overridden via CLI argument.
#
# Skips already-completed runs (checks for Unimal-v0.pt).
#
# Usage: bash scripts/train_loki_all_cluster_tasks.sh [num_gpus] [max_concurrent_per_gpu] [test_mode]
#
# Examples:
#   bash scripts/train_loki_all_cluster_tasks.sh 2          # 2 GPUs, auto-detect concurrency
#   bash scripts/train_loki_all_cluster_tasks.sh 2 10       # 2 GPUs, force 10 per GPU
#   bash scripts/train_loki_all_cluster_tasks.sh 2 0 1      # 2 GPUs, auto-detect, test mode (1e5 steps)

NUM_GPUS=${1:-2}
MAX_CONCURRENT_OVERRIDE=${2:-0}      # 0 = auto-detect from GPU memory tier
TEST_MODE=${3:-0}                    # 1 = reduced steps for testing (1e5 instead of 1e8)

STAGGER_TIME=300                     # 5 minutes between launches
RAM_THRESHOLD=80000                  # Minimum free system RAM in MiB (80 GB)

NUM_WALKER=20
NUM_CLUSTERS=20
RNG_SEED=3429
TASKS=("locomotion" "obstacle" "incline")

DROP_FREQ=2
NUM_DROP=2

# Scheduling state
GPU_PER_JOB=0                        # Measured per-job GPU memory (MiB) — set after Phase 1
RAM_PER_JOB=0                        # Measured per-job RAM (MiB) — set after Phase 1
NEXT_GPU=0                           # Round-robin counter for GPU assignment
FAILURE_COOLDOWN_UNTIL=0             # Epoch seconds: block launches until this time
MAX_CONCURRENT_PER_GPU=0             # Set during initialization

# Per-GPU job count tracking
declare -A GPU_JOB_COUNT
declare -A PID_TO_GPU
declare -A PID_TO_JOB
declare -A PID_TO_START
CHILD_PIDS=()

# Job queue
QUEUE_TASKS=()
QUEUE_CLUSTERS=()
QUEUE_IDX=0

# Counters
TOTAL_JOBS=0
SKIPPED=0
COMPLETED=0
LAUNCHED=0

# Initialize per-GPU counters
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_JOB_COUNT[$gpu]=0
done

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
    # Use results.json as the completion marker — it is only written after train() finishes
    # and save_rewards() runs.  Unimal-v0.pt is saved periodically during training so it
    # cannot distinguish a partial run from a completed one.
    local results="metamorph/output/loki/$env_type/kmeans_cluster/$NUM_CLUSTERS/$cluster/walker$NUM_WALKER/freq$DROP_FREQ/drop$NUM_DROP/seed$RNG_SEED/Unimal-v0_results.json"
    [ -f "$results" ]
}

get_gpu_free_mem() {
    local gpu_idx=$1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$gpu_idx 2>/dev/null | tr -d ' '
}

get_gpu_total_mem() {
    local gpu_idx=$1
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$gpu_idx 2>/dev/null | tr -d ' '
}

get_free_gpu_mem() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}'
}

get_total_gpu_mem() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}'
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

# Determine max concurrent evaluations per GPU based on VRAM tier
detect_max_concurrent_per_gpu() {
    local gpu_mem=$(get_gpu_total_mem 0)
    if [ "$gpu_mem" -le 50000 ]; then
        # ~40 GB tier (e.g., A100-40GB)
        MAX_CONCURRENT_PER_GPU=5
    elif [ "$gpu_mem" -le 100000 ]; then
        # ~80 GB tier (e.g., A100-80GB)
        MAX_CONCURRENT_PER_GPU=10
    else
        # ~141 GB tier (e.g., H200)
        MAX_CONCURRENT_PER_GPU=16
    fi
    echo "[CONFIG] GPU 0 has ${gpu_mem} MiB VRAM -> max ${MAX_CONCURRENT_PER_GPU} concurrent per GPU"
}

# Count running jobs, prune dead PIDs, and apply failure cooldowns.
# IMPORTANT: Sets global RUNNING_COUNT instead of echoing, to avoid subshell issues
# when called via $(count_running_jobs). All state changes (GPU_JOB_COUNT, CHILD_PIDS,
# COMPLETED, FAILURE_COOLDOWN_UNTIL) persist in the parent shell.
RUNNING_COUNT=0
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
            local gpu="${PID_TO_GPU[$pid]}"
            local job="${PID_TO_JOB[$pid]}"
            local start="${PID_TO_START[$pid]}"
            local runtime=0
            if [ -n "$start" ]; then
                runtime=$(( now - start ))
            fi

            # Decrement GPU job count
            if [ -n "$gpu" ]; then
                GPU_JOB_COUNT[$gpu]=$(( ${GPU_JOB_COUNT[$gpu]} - 1 ))
                if [ "${GPU_JOB_COUNT[$gpu]}" -lt 0 ]; then
                    GPU_JOB_COUNT[$gpu]=0
                fi
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
            unset PID_TO_GPU[$pid]
        fi
    done
    CHILD_PIDS=("${alive_pids[@]}")
}

# Advance the round-robin GPU counter and set TARGET_GPU.
# IMPORTANT: Sets global TARGET_GPU instead of echoing, to avoid subshell issues.
TARGET_GPU=0
advance_gpu_rr() {
    TARGET_GPU=$((NEXT_GPU % NUM_GPUS))
    NEXT_GPU=$(( (NEXT_GPU + 1) % NUM_GPUS ))
}

# Launch the next job from the queue on the given GPU
launch_next_job() {
    local gpu=$1

    if [ "$QUEUE_IDX" -ge "${#QUEUE_TASKS[@]}" ]; then
        return 1  # No more jobs
    fi

    local task="${QUEUE_TASKS[$QUEUE_IDX]}"
    local cluster="${QUEUE_CLUSTERS[$QUEUE_IDX]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))

    local job_desc="task=${task} cluster=${cluster} (GPU=${gpu})"

    if [ "$TEST_MODE" -eq 1 ]; then
        bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $cluster $RNG_SEED $task $gpu 1 &
    else
        bash scripts/train_loki_task.sh $NUM_WALKER $NUM_CLUSTERS $cluster $RNG_SEED $task $gpu &
    fi
    local pid=$!
    CHILD_PIDS+=($pid)
    PID_TO_JOB[$pid]="$job_desc"
    PID_TO_START[$pid]=$(date +%s)
    PID_TO_GPU[$pid]=$gpu

    GPU_JOB_COUNT[$gpu]=$(( ${GPU_JOB_COUNT[$gpu]} + 1 ))

    echo "[LAUNCH] $job_desc (PID=$pid) [GPU $gpu: ${GPU_JOB_COUNT[$gpu]} jobs]"
    LAUNCHED=$((LAUNCHED + 1))

    return 0
}

# Measure per-job resource consumption (called after Phase 1)
measure_per_job_usage() {
    update_running_jobs
    if [ "$RUNNING_COUNT" -eq 0 ]; then
        echo "[WARNING] No running jobs to measure."
        return 1
    fi

    local total_gpu_used=0
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        local total=$(get_gpu_total_mem $gpu)
        local free=$(get_gpu_free_mem $gpu)
        local used=$((total - free))
        total_gpu_used=$((total_gpu_used + used))
    done

    local total_ram=$(get_total_ram)
    local free_ram=$(get_free_ram)
    local used_ram=$((total_ram - free_ram))

    GPU_PER_JOB=$((total_gpu_used / RUNNING_COUNT))
    RAM_PER_JOB=$((used_ram / RUNNING_COUNT))

    echo "[PROFILING] Per-job resource estimate: ~${GPU_PER_JOB} MiB GPU, ~${RAM_PER_JOB} MiB RAM"
}

# Display comprehensive job status
print_status() {
    update_running_jobs
    local queued=$(( ${#QUEUE_TASKS[@]} - QUEUE_IDX ))
    local now=$(date +%s)
    local total_max=$(( MAX_CONCURRENT_PER_GPU * NUM_GPUS ))

    echo ""
    echo "=== Job Status (${RUNNING_COUNT}/${total_max} running | ${queued} queued | ${COMPLETED} completed | ${SKIPPED} skipped) ==="

    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            local elapsed=$(( now - ${PID_TO_START[$pid]} ))
            echo "  [RUNNING]  ${PID_TO_JOB[$pid]}  PID=$pid  ($(format_elapsed $elapsed))"
        fi
    done

    if [ "$QUEUE_IDX" -lt "${#QUEUE_TASKS[@]}" ]; then
        echo "  [QUEUED]   task=${QUEUE_TASKS[$QUEUE_IDX]} cluster=${QUEUE_CLUSTERS[$QUEUE_IDX]}  (next)"
    fi

    local free_gpu=$(get_free_gpu_mem)
    local total_gpu=$(get_total_gpu_mem)
    local free_ram=$(get_free_ram)
    local total_ram=$(get_total_ram)
    local used_gpu=$((total_gpu - free_gpu))
    local used_ram=$((total_ram - free_ram))

    echo "  Resources: GPU ${used_gpu}/${total_gpu} MiB used | RAM ${used_ram}/${total_ram} MiB used (free: ${free_ram} MiB, threshold: ${RAM_THRESHOLD} MiB)"
    if [ "$GPU_PER_JOB" -gt 0 ]; then
        echo "  Per-job estimate: ~${GPU_PER_JOB} MiB GPU, ~${RAM_PER_JOB} MiB RAM"
    fi

    echo "  --- Per-GPU Usage ---"
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader --id=$gpu 2>/dev/null)
        echo "    GPU $gpu: ${gpu_info} | Jobs: ${GPU_JOB_COUNT[$gpu]}"
    done

    if [ "$now" -lt "$FAILURE_COOLDOWN_UNTIL" ]; then
        local cd_remaining=$(( FAILURE_COOLDOWN_UNTIL - now ))
        echo "  [COOLDOWN] ${cd_remaining}s remaining after failure"
    fi
    echo "================================================================"
    echo ""
}

# --- Build job queue ---

echo "=== LOKI Simplified Sequential Scheduler ==="
echo "Num GPUs: $NUM_GPUS"
echo "Max concurrent override: ${MAX_CONCURRENT_OVERRIDE} (0 = auto-detect)"
echo "Stagger time: ${STAGGER_TIME}s (5 min)"
echo "RAM threshold: ${RAM_THRESHOLD} MiB"
echo "Test mode: ${TEST_MODE}"
echo ""

# Detect or set max concurrent per GPU
if [ "$MAX_CONCURRENT_OVERRIDE" -gt 0 ]; then
    MAX_CONCURRENT_PER_GPU=$MAX_CONCURRENT_OVERRIDE
    echo "[CONFIG] Using override: ${MAX_CONCURRENT_PER_GPU} concurrent per GPU"
else
    detect_max_concurrent_per_gpu
fi

TOTAL_MAX=$(( MAX_CONCURRENT_PER_GPU * NUM_GPUS ))
echo "[CONFIG] Total max concurrent: ${TOTAL_MAX} (${MAX_CONCURRENT_PER_GPU} per GPU x ${NUM_GPUS} GPUs)"
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

# --- Phase 1: Staggered initial launch (one per GPU) ---

echo ""
echo "=== Phase 1: Staggered initial launch (one per GPU, ${STAGGER_TIME}s between each) ==="

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    if [ "$QUEUE_IDX" -ge "${#QUEUE_TASKS[@]}" ]; then
        echo "[PHASE 1] Queue exhausted after ${gpu} GPUs"
        break
    fi

    launch_next_job $gpu

    # Wait between launches (skip wait after last GPU in phase 1)
    if [ "$gpu" -lt $((NUM_GPUS - 1)) ] && [ "$QUEUE_IDX" -lt "${#QUEUE_TASKS[@]}" ]; then
        echo "[PHASE 1] Waiting ${STAGGER_TIME}s before launching on next GPU..."
        elapsed=0
        while [ "$elapsed" -lt "$STAGGER_TIME" ]; do
            sleep_chunk=60
            remaining=$((STAGGER_TIME - elapsed))
            if [ "$sleep_chunk" -gt "$remaining" ]; then
                sleep_chunk=$remaining
            fi
            sleep $sleep_chunk
            elapsed=$((elapsed + sleep_chunk))

            # Check if job is still alive
            update_running_jobs
            if [ "$RUNNING_COUNT" -eq 0 ]; then
                echo "[WARNING] All jobs exited during Phase 1 stagger. Check logs."
            fi
            echo "  Phase 1 stagger: ${elapsed}/${STAGGER_TIME}s (${RUNNING_COUNT} jobs running)"
        done
    fi
done

# Wait for last Phase 1 job to stabilize
echo "[PHASE 1] Waiting ${STAGGER_TIME}s for last job to stabilize..."
elapsed=0
while [ "$elapsed" -lt "$STAGGER_TIME" ]; do
    sleep_chunk=60
    remaining=$((STAGGER_TIME - elapsed))
    if [ "$sleep_chunk" -gt "$remaining" ]; then
        sleep_chunk=$remaining
    fi
    sleep $sleep_chunk
    elapsed=$((elapsed + sleep_chunk))
    update_running_jobs
    echo "  Phase 1 stabilize: ${elapsed}/${STAGGER_TIME}s (${RUNNING_COUNT} jobs running)"
done

# Measure per-job resource usage
measure_per_job_usage
print_status

# --- Phase 2: Sequential scheduling with resource gating ---

echo ""
echo "=== Phase 2: Sequential scheduling (max ${TOTAL_MAX} concurrent, ${STAGGER_TIME}s stagger, RAM threshold ${RAM_THRESHOLD} MiB) ==="

while true; do
    update_running_jobs
    pending_left=$(( ${#QUEUE_TASKS[@]} - QUEUE_IDX ))

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
        if [ "$RUNNING_COUNT" -ge "$TOTAL_MAX" ]; then
            echo "[WAIT] At max concurrent limit (${RUNNING_COUNT}/${TOTAL_MAX}). Waiting for jobs to finish..."
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

        # All conditions met — launch next job on next GPU (round-robin)
        advance_gpu_rr
        launch_next_job $TARGET_GPU
        print_status

        # Stagger: wait 5 minutes before scheduling next
        echo "[STAGGER] Waiting ${STAGGER_TIME}s before next launch..."
        elapsed=0
        while [ "$elapsed" -lt "$STAGGER_TIME" ]; do
            sleep_chunk=60
            remaining=$((STAGGER_TIME - elapsed))
            if [ "$sleep_chunk" -gt "$remaining" ]; then
                sleep_chunk=$remaining
            fi
            sleep $sleep_chunk
            elapsed=$((elapsed + sleep_chunk))
            update_running_jobs
            echo "  Stagger: ${elapsed}/${STAGGER_TIME}s (${RUNNING_COUNT} jobs running)"
        done
        continue
    fi

    # No pending jobs, just wait for running jobs to finish
    print_status
    sleep 60
done

# --- Done ---

echo ""
echo "=== All jobs completed ==="
echo "Total: $TOTAL_JOBS | Completed: $COMPLETED | Skipped: $SKIPPED | Launched: $LAUNCHED"

trap - EXIT
