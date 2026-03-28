#!/bin/bash
#SBATCH --job-name=loki-500k-eval
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=log/slurm/loki-500k-eval-%j.out
#SBATCH --error=log/slurm/loki-500k-eval-%j.err

# === 500K Full Evaluation Launcher ===
#
# Trains LOKI on 500K-clustered morphologies across user-specified tasks and clusters.
# Each (task, cluster) pair is launched as a background job with staggered starts.
#
# ============================================================
# CONFIGURE THESE BEFORE EACH SUBMISSION:
# ============================================================

# Tasks to evaluate (pick from: locomotion obstacle many_obstacle bump incline push_box_incline exploration patrol manipulation_ball)
TASKS=(locomotion obstacle bump)

# Clusters to evaluate (0-39 for 40 clusters; adjust range as needed)
CLUSTERS=($(seq 0 39))

# Max concurrent jobs on a single GPU (adjust by GPU VRAM)
MAX_CONCURRENT_PER_GPU=4

# Stagger time between launches (seconds)
STAGGER_TIME=300

# ============================================================
# END OF USER CONFIGURATION
# ============================================================

NUM_WALKER=20
NUM_CLUSTERS=40
RNG_SEED=3429

# Setup environment
module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

mkdir -p log/slurm log/train_loki_task_500k

# Detect GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1

echo "=== LOKI 500K Full Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $NUM_GPUS"
echo "Tasks: ${TASKS[*]}"
echo "Clusters: ${CLUSTERS[0]}..${CLUSTERS[-1]} (${#CLUSTERS[@]} total)"
echo "Max concurrent per GPU: $MAX_CONCURRENT_PER_GPU"
echo "Total jobs: $(( ${#TASKS[@]} * ${#CLUSTERS[@]} ))"
echo "Start time: $(date)"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Build job queue: (task, cluster) pairs
QUEUE_TASKS=()
QUEUE_CLUSTERS=()
SKIPPED=0

for task in "${TASKS[@]}"; do
    for cluster in "${CLUSTERS[@]}"; do
        # Map task name to ENV_TYPE for output path check
        case "$task" in
            locomotion) env_type="ft" ;;
            *)          env_type="$task" ;;
        esac

        # Skip already-completed jobs (results.json exists)
        CKPT_PATH="./metamorph/output/loki_500k/$env_type/kmeans_cluster/$NUM_CLUSTERS/$cluster/walker$NUM_WALKER/freq2/drop2/seed$RNG_SEED"
        if [ -f "$CKPT_PATH/Unimal-v0_results.json" ]; then
            echo "[SKIP] $task cluster $cluster — already completed"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        QUEUE_TASKS+=("$task")
        QUEUE_CLUSTERS+=("$cluster")
    done
done

TOTAL_JOBS=${#QUEUE_TASKS[@]}
echo ""
echo "Jobs to run: $TOTAL_JOBS (skipped $SKIPPED already-completed)"
echo ""

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "Nothing to do — all jobs already completed."
    exit 0
fi

# ============================================================
# Phase 1: Staggered launch — 1 job per GPU to measure baseline
# ============================================================
echo "=== Phase 1: Staggered Launch (1 per GPU, ${STAGGER_TIME}s stagger) ==="

declare -A GPU_JOB_COUNT    # GPU_ID -> number of active jobs
declare -A PID_TO_GPU       # PID -> GPU_ID
declare -A PID_TO_JOB       # PID -> "task:cluster"
PIDS=()

for (( gpu=0; gpu<NUM_GPUS && gpu<TOTAL_JOBS; gpu++ )); do
    task="${QUEUE_TASKS[$gpu]}"
    cluster="${QUEUE_CLUSTERS[$gpu]}"

    echo "[$(date)] [Phase1] Launching $task cluster $cluster on GPU $gpu"
    bash scripts/train_loki_task_500k.sh $NUM_WALKER $NUM_CLUSTERS $cluster $RNG_SEED $task $gpu &
    pid=$!
    PIDS+=($pid)
    PID_TO_GPU[$pid]=$gpu
    PID_TO_JOB[$pid]="$task:$cluster"
    GPU_JOB_COUNT[$gpu]=$(( ${GPU_JOB_COUNT[$gpu]:-0} + 1 ))
    echo "[$(date)] [Phase1] $task cluster $cluster launched (PID=$pid, GPU=$gpu)"

    if [ "$gpu" -lt $(( NUM_GPUS - 1 )) ] && [ "$gpu" -lt $(( TOTAL_JOBS - 1 )) ]; then
        echo "[$(date)] Waiting ${STAGGER_TIME}s before next Phase 1 launch..."
        sleep $STAGGER_TIME
    fi
done

NEXT_IDX=$NUM_GPUS

# ============================================================
# Phase 2: Fill remaining jobs with resource gating
# ============================================================
echo ""
echo "=== Phase 2: Resource-Gated Launch (max $MAX_CONCURRENT_PER_GPU per GPU) ==="
echo "Remaining jobs: $(( TOTAL_JOBS - NEXT_IDX ))"
echo ""

launch_next_job() {
    local gpu=$1
    if [ "$NEXT_IDX" -ge "$TOTAL_JOBS" ]; then
        return 1
    fi

    local task="${QUEUE_TASKS[$NEXT_IDX]}"
    local cluster="${QUEUE_CLUSTERS[$NEXT_IDX]}"
    NEXT_IDX=$((NEXT_IDX + 1))

    echo "[$(date)] [Phase2] Launching $task cluster $cluster on GPU $gpu"
    bash scripts/train_loki_task_500k.sh $NUM_WALKER $NUM_CLUSTERS $cluster $RNG_SEED $task $gpu &
    local pid=$!
    PIDS+=($pid)
    PID_TO_GPU[$pid]=$gpu
    PID_TO_JOB[$pid]="$task:$cluster"
    GPU_JOB_COUNT[$gpu]=$(( ${GPU_JOB_COUNT[$gpu]:-0} + 1 ))
    echo "[$(date)] [Phase2] $task cluster $cluster launched (PID=$pid, GPU=$gpu)"

    sleep 60  # brief stagger between Phase 2 launches
    return 0
}

# Main scheduling loop: wait for any child to finish, then launch replacement
COMPLETED=0
FAILED=0

while [ "$COMPLETED" -lt "$TOTAL_JOBS" ]; do
    # Try to fill GPU slots
    for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
        while [ "${GPU_JOB_COUNT[$gpu]:-0}" -lt "$MAX_CONCURRENT_PER_GPU" ] && [ "$NEXT_IDX" -lt "$TOTAL_JOBS" ]; do
            # Check system RAM (skip if below 80GB free)
            FREE_RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}')
            if [ -n "$FREE_RAM_GB" ] && [ "$FREE_RAM_GB" -lt 80 ]; then
                echo "[$(date)] Low RAM (${FREE_RAM_GB}GB free) — waiting before launching more"
                break 2
            fi
            launch_next_job $gpu || break
        done
    done

    # Wait for any child process to finish
    wait -n 2>/dev/null
    EXIT_CODE=$?

    # Find which PID(s) finished
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            # This PID has exited
            job="${PID_TO_JOB[$pid]}"
            gpu="${PID_TO_GPU[$pid]}"

            if [ -n "$job" ]; then
                wait "$pid" 2>/dev/null
                exit_code=$?
                COMPLETED=$((COMPLETED + 1))
                GPU_JOB_COUNT[$gpu]=$(( ${GPU_JOB_COUNT[$gpu]} - 1 ))

                if [ "$exit_code" -eq 0 ]; then
                    echo "[$(date)] DONE $job (PID=$pid, GPU=$gpu) [$COMPLETED/$TOTAL_JOBS]"
                else
                    echo "[$(date)] FAIL $job (PID=$pid, GPU=$gpu, exit=$exit_code) [$COMPLETED/$TOTAL_JOBS]"
                    FAILED=$((FAILED + 1))
                fi

                # Clear from tracking
                unset PID_TO_JOB[$pid]
                unset PID_TO_GPU[$pid]
            fi
        fi
    done

    sleep 5
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== 500K Evaluation Complete ==="
echo "Total: $TOTAL_JOBS | Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED"
echo "End time: $(date)"
