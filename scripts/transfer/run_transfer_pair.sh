#!/bin/bash
# =============================================================================
# Generic Cross-Task Transfer Pipeline (HPC/SLURM)
# =============================================================================
#
# Runs the full bidirectional transfer experiment between any two tasks:
#   Phase 0: Inventory — find which clusters have trained checkpoints
#   Phase 1: Zero-shot evaluation (A→B and B→A)
#   Phase 2: Fine-tuning budget sweep (A→B and B→A, budget-first ordering)
#
# Reusable for ANY task pair. All task-specific logic is in run_single_transfer.sh.
#
# SLURM submission:
#   sbatch --gres=gpu:h200:2 --cpus-per-task=16 --mem=256G --time=1-00:00:00 \
#          --output=log/slurm/transfer-%j.out --error=log/slurm/transfer-%j.err \
#          scripts/transfer/run_transfer_pair.sh ft incline
#
# Interactive:
#   bash scripts/transfer/run_transfer_pair.sh ft incline
#   bash scripts/transfer/run_transfer_pair.sh obstacle bump 2 4
#   bash scripts/transfer/run_transfer_pair.sh ft patrol 0 0 40
#
# Arguments:
#   $1  task_a       First task  (required, e.g. "ft", "obstacle", "bump")
#   $2  task_b       Second task (required, e.g. "incline", "many_obstacle")
#   $3  num_gpus     Number of GPUs (default: auto-detect)
#   $4  max_ft_gpu   Max concurrent fine-tune per GPU (default: auto-detect)
#   $5  num_clusters Total clusters (default: 20)
#   $6  seed         Random seed (default: 3429)
# =============================================================================

#SBATCH --job-name=loki-transfer
#SBATCH --partition=work1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=log/slurm/transfer-%j.out
#SBATCH --error=log/slurm/transfer-%j.err

set -e

# ==================== Arguments ====================
TASK_A="${1:?Usage: $0 <task_a> <task_b> [num_gpus] [max_ft_gpu] [num_clusters] [seed]}"
TASK_B="${2:?Usage: $0 <task_a> <task_b> [num_gpus] [max_ft_gpu] [num_clusters] [seed]}"
NUM_GPUS=${3:-0}
MAX_FT_OVERRIDE=${4:-0}
NUM_CLUSTERS=${5:-20}
SEED=${6:-3429}

NUM_WALKER=20
DROP_FREQ=2
NUM_DROP=2

BUDGETS=("2e6" "5e6" "1e7" "2e7" "5e7")
ZERO_SHOT_EPISODES=50

# Scheduling
STAGGER_TIME_ZS=10
STAGGER_TIME_FT=30
RAM_THRESHOLD=40000

# Logging
PIPELINE_START=$(date +%s)
PAIR_LABEL="${TASK_A}_x_${TASK_B}"
PHASE_LOG_DIR="log/transfer/pipeline_${PAIR_LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PHASE_LOG_DIR"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
elapsed_since() { echo "$(( $(date +%s) - $1 ))s"; }

# ==================== Environment ====================
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running as SLURM job $SLURM_JOB_ID on $(hostname)"
    module load cuda/12.3 2>/dev/null || true
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate loki 2>/dev/null || true
fi

export MUJOCO_GL=egl
export WANDB_PROJECT="LOKI-transfer"
export WANDB_MODE=disabled

# ==================== GPU Detection ====================
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    [ "$NUM_GPUS" -eq 0 ] && NUM_GPUS=1
fi

get_gpu_total_mem() {
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$1 2>/dev/null | tr -d ' '
}

get_free_ram() {
    free -m 2>/dev/null | awk '/^Mem:/ {print $7}'
}

detect_concurrency() {
    local gpu_mem=$(get_gpu_total_mem 0)
    if [ "$gpu_mem" -le 50000 ]; then
        MAX_ZS_PER_GPU=5;  MAX_FT_PER_GPU=2
    elif [ "$gpu_mem" -le 100000 ]; then
        MAX_ZS_PER_GPU=10; MAX_FT_PER_GPU=3
    else
        MAX_ZS_PER_GPU=16; MAX_FT_PER_GPU=4
    fi
}

MAX_ZS_PER_GPU=0
MAX_FT_PER_GPU=0
detect_concurrency
if [ "$MAX_FT_OVERRIDE" -gt 0 ]; then
    MAX_FT_PER_GPU=$MAX_FT_OVERRIDE
fi

TOTAL_MAX=$((NUM_GPUS * MAX_ZS_PER_GPU))  # starts with Phase 1 limit

echo "================================================================"
echo "  Transfer Pipeline: ${TASK_A} <-> ${TASK_B}"
echo "================================================================"
echo "  GPUs: ${NUM_GPUS}"
echo "  Phase 1 (zero-shot): max ${MAX_ZS_PER_GPU}/GPU = $((NUM_GPUS * MAX_ZS_PER_GPU)) total"
echo "  Phase 2 (fine-tune): max ${MAX_FT_PER_GPU}/GPU = $((NUM_GPUS * MAX_FT_PER_GPU)) total"
echo "  Clusters: ${NUM_CLUSTERS}, Seed: ${SEED}"
echo "  Budgets: ${BUDGETS[*]}"
echo "  Logs: ${PHASE_LOG_DIR}/"
echo "================================================================"
echo ""

# ==================== Scheduling Helpers ====================
NEXT_GPU=0
declare -A GPU_JOB_COUNT
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_JOB_COUNT[$gpu]=0
done

RUNNING_PIDS=()
TOTAL_LAUNCHED=0

get_task_dir() {
    case "$1" in
        locomotion|ft) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

has_source_ckpt() {
    local task=$1 cluster=$2
    local dir=$(get_task_dir "$task")
    [ -f "metamorph/output/loki/${dir}/kmeans_cluster/${NUM_CLUSTERS}/${cluster}/walker${NUM_WALKER}/freq${DROP_FREQ}/drop${NUM_DROP}/seed${SEED}/Unimal-v0.pt" ]
}

update_running() {
    local new_pids=()
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        fi
    done
    RUNNING_PIDS=("${new_pids[@]}")
}

wait_for_slot() {
    while true; do
        update_running
        if [ "${#RUNNING_PIDS[@]}" -lt "$TOTAL_MAX" ]; then
            local free_ram=$(get_free_ram)
            if [ "$free_ram" -gt "$RAM_THRESHOLD" ]; then
                break
            fi
            echo "[WAIT] RAM: ${free_ram}M < ${RAM_THRESHOLD}M (${#RUNNING_PIDS[@]} running)"
        fi
        sleep 5
    done
}

launch() {
    local mode=$1 src=$2 tgt=$3 cluster=$4 budget=${5:-""}

    wait_for_slot

    local gpu=$((NEXT_GPU % NUM_GPUS))
    NEXT_GPU=$(( (NEXT_GPU + 1) % NUM_GPUS ))

    local desc="${src}->${tgt} C${cluster}"
    [ -n "$budget" ] && desc="${desc} ${budget}"

    update_running
    echo "[$(ts)] [LAUNCH] GPU ${gpu}: ${desc} (${#RUNNING_PIDS[@]}/${TOTAL_MAX} slots)"

    CUDA_VISIBLE_DEVICES=$gpu \
    bash scripts/transfer/run_single_transfer.sh \
        "$mode" "$src" "$tgt" "$cluster" "$SEED" "$NUM_CLUSTERS" \
        "$budget" "$ZERO_SHOT_EPISODES" &

    RUNNING_PIDS+=($!)
    GPU_JOB_COUNT[$gpu]=$(( ${GPU_JOB_COUNT[$gpu]} + 1 ))
    TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

    if [ "$mode" = "zero_shot" ]; then
        sleep $STAGGER_TIME_ZS
    else
        sleep $STAGGER_TIME_FT
    fi
}

cleanup() {
    echo ""
    echo "[CLEANUP] Killing child processes..."
    for pid in "${RUNNING_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    exit 1
}
trap cleanup SIGTERM SIGINT SIGHUP

# ==========================================================================
# PHASE 0: Inventory
# ==========================================================================
PHASE0_LOG="${PHASE_LOG_DIR}/phase0_inventory.log"

echo "================================================================" | tee "$PHASE0_LOG"
echo "[$(ts)] PHASE 0: Inventory for ${TASK_A} <-> ${TASK_B}"          | tee -a "$PHASE0_LOG"
echo "================================================================" | tee -a "$PHASE0_LOG"

AVAIL_A=()
AVAIL_B=()

for C in $(seq 0 $((NUM_CLUSTERS - 1))); do
    if has_source_ckpt "$TASK_A" "$C"; then
        AVAIL_A+=($C)
    fi
    if has_source_ckpt "$TASK_B" "$C"; then
        AVAIL_B+=($C)
    fi
done

echo "  ${TASK_A} checkpoints: ${#AVAIL_A[@]}/${NUM_CLUSTERS} [${AVAIL_A[*]}]" | tee -a "$PHASE0_LOG"
echo "  ${TASK_B} checkpoints: ${#AVAIL_B[@]}/${NUM_CLUSTERS} [${AVAIL_B[*]}]" | tee -a "$PHASE0_LOG"
echo ""                                                                         | tee -a "$PHASE0_LOG"
echo "  Jobs planned:"                                                          | tee -a "$PHASE0_LOG"
echo "    Phase 1 (zero-shot):  $((${#AVAIL_A[@]} + ${#AVAIL_B[@]}))"          | tee -a "$PHASE0_LOG"
echo "    Phase 2 (fine-tune):  $(( (${#AVAIL_A[@]} + ${#AVAIL_B[@]}) * ${#BUDGETS[@]} ))" | tee -a "$PHASE0_LOG"
echo ""

if [ ${#AVAIL_A[@]} -eq 0 ] && [ ${#AVAIL_B[@]} -eq 0 ]; then
    echo "ERROR: No source checkpoints found. Exiting." | tee -a "$PHASE0_LOG"
    exit 1
fi

# ==========================================================================
# PHASE 1: Zero-shot evaluation
# ==========================================================================
PHASE1_START=$(date +%s)
PHASE1_LOG="${PHASE_LOG_DIR}/phase1_zero_shot.log"
PHASE1_EXPECTED=$((${#AVAIL_A[@]} + ${#AVAIL_B[@]}))

echo "================================================================"              | tee "$PHASE1_LOG"
echo "[$(ts)] PHASE 1: Zero-shot (${PHASE1_EXPECTED} jobs)"                          | tee -a "$PHASE1_LOG"
echo "  Output: metamorph/output/transfer/zero_shot/{pair}/c{C}/seed${SEED}/"        | tee -a "$PHASE1_LOG"
echo "  Result: eval_results.json"                                                   | tee -a "$PHASE1_LOG"
echo "================================================================"              | tee -a "$PHASE1_LOG"

# A -> B
if [ ${#AVAIL_A[@]} -gt 0 ]; then
    echo "--- ${TASK_A} -> ${TASK_B} (${#AVAIL_A[@]} clusters) ---" | tee -a "$PHASE1_LOG"
    for C in "${AVAIL_A[@]}"; do
        launch zero_shot "$TASK_A" "$TASK_B" "$C"
    done
fi

# B -> A
if [ ${#AVAIL_B[@]} -gt 0 ]; then
    echo "--- ${TASK_B} -> ${TASK_A} (${#AVAIL_B[@]} clusters) ---" | tee -a "$PHASE1_LOG"
    for C in "${AVAIL_B[@]}"; do
        launch zero_shot "$TASK_B" "$TASK_A" "$C"
    done
fi

echo "[$(ts)] Waiting for ${TOTAL_LAUNCHED} zero-shot jobs..." | tee -a "$PHASE1_LOG"
wait

# Phase 1 completion check
P1_DONE=0
P1_FAIL=0
P1_RESULTS=""

check_zero_shot() {
    local src=$1 tgt=$2
    shift 2
    local clusters=("$@")
    for C in "${clusters[@]}"; do
        local F="metamorph/output/transfer/zero_shot/${src}_to_${tgt}/c${C}/seed${SEED}/eval_results.json"
        if [ -f "$F" ]; then
            local MEAN=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"mean_reward\"]:.1f}')" 2>/dev/null || echo "?")
            local STD=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"std_reward\"]:.1f}')" 2>/dev/null || echo "?")
            echo "  [DONE] ${src}->${tgt} C${C}: ${MEAN} +/- ${STD}" | tee -a "$PHASE1_LOG"
            P1_RESULTS="${P1_RESULTS}  ${src}->${tgt} C${C}: ${MEAN} +/- ${STD}\n"
            P1_DONE=$((P1_DONE + 1))
        else
            echo "  [FAIL] ${src}->${tgt} C${C}" | tee -a "$PHASE1_LOG"
            P1_FAIL=$((P1_FAIL + 1))
        fi
    done
}

check_zero_shot "$TASK_A" "$TASK_B" "${AVAIL_A[@]}"
check_zero_shot "$TASK_B" "$TASK_A" "${AVAIL_B[@]}"

echo ""                                                                         | tee -a "$PHASE1_LOG"
echo "[$(ts)] PHASE 1 SUMMARY: ${P1_DONE}/${PHASE1_EXPECTED} done (${P1_FAIL} failed) in $(elapsed_since $PHASE1_START)" | tee -a "$PHASE1_LOG"
echo -e "$P1_RESULTS"                                                           | tee -a "$PHASE1_LOG"

# Reset for Phase 2
PHASE1_TOTAL=$TOTAL_LAUNCHED
TOTAL_LAUNCHED=0
RUNNING_PIDS=()
TOTAL_MAX=$((NUM_GPUS * MAX_FT_PER_GPU))

# ==========================================================================
# PHASE 2: Fine-tuning (budget-first ordering for better concurrency)
# ==========================================================================
PHASE2_START=$(date +%s)
PHASE2_LOG="${PHASE_LOG_DIR}/phase2_finetune.log"
PHASE2_EXPECTED=$(( (${#AVAIL_A[@]} + ${#AVAIL_B[@]}) * ${#BUDGETS[@]} ))

echo "================================================================"              | tee "$PHASE2_LOG"
echo "[$(ts)] PHASE 2: Fine-tuning (${PHASE2_EXPECTED} jobs)"                        | tee -a "$PHASE2_LOG"
echo "  Concurrency: ${MAX_FT_PER_GPU}/GPU = ${TOTAL_MAX} total"                    | tee -a "$PHASE2_LOG"
echo "  Output: metamorph/output/transfer/finetune/{pair}/c{C}/steps_{B}/seed${SEED}/" | tee -a "$PHASE2_LOG"
echo "  Result: Unimal-v0_results.json"                                              | tee -a "$PHASE2_LOG"
echo "================================================================"              | tee -a "$PHASE2_LOG"

# Budget-first ordering: all clusters get cheap jobs first, expensive last.
# This prevents long 5e7 jobs from blocking slots while short jobs queue.
for B in "${BUDGETS[@]}"; do
    echo "--- Budget ${B} ---" | tee -a "$PHASE2_LOG"

    # A -> B
    for C in "${AVAIL_A[@]}"; do
        echo "[$(ts)] [LAUNCH] ${TASK_A}->${TASK_B} C${C} ${B}" | tee -a "$PHASE2_LOG"
        launch finetune "$TASK_A" "$TASK_B" "$C" "$B"
    done

    # B -> A
    if [ ${#AVAIL_B[@]} -gt 0 ]; then
        for C in "${AVAIL_B[@]}"; do
            echo "[$(ts)] [LAUNCH] ${TASK_B}->${TASK_A} C${C} ${B}" | tee -a "$PHASE2_LOG"
            launch finetune "$TASK_B" "$TASK_A" "$C" "$B"
        done
    fi
done

echo "[$(ts)] Waiting for ${TOTAL_LAUNCHED} fine-tune jobs..." | tee -a "$PHASE2_LOG"
wait

# Phase 2 completion check
P2_DONE=0
P2_FAIL=0

check_finetune() {
    local src=$1 tgt=$2
    shift 2
    local clusters=("$@")
    for C in "${clusters[@]}"; do
        for B in "${BUDGETS[@]}"; do
            local F="metamorph/output/transfer/finetune/${src}_to_${tgt}/c${C}/steps_${B}/seed${SEED}/Unimal-v0_results.json"
            if [ -f "$F" ]; then
                echo "  [DONE] ${src}->${tgt} C${C} ${B}" | tee -a "$PHASE2_LOG"
                P2_DONE=$((P2_DONE + 1))
            else
                echo "  [FAIL] ${src}->${tgt} C${C} ${B}" | tee -a "$PHASE2_LOG"
                P2_FAIL=$((P2_FAIL + 1))
            fi
        done
    done
}

check_finetune "$TASK_A" "$TASK_B" "${AVAIL_A[@]}"
check_finetune "$TASK_B" "$TASK_A" "${AVAIL_B[@]}"

echo ""                                                                         | tee -a "$PHASE2_LOG"
echo "[$(ts)] PHASE 2 SUMMARY: ${P2_DONE}/${PHASE2_EXPECTED} done (${P2_FAIL} failed) in $(elapsed_since $PHASE2_START)" | tee -a "$PHASE2_LOG"

# ==========================================================================
# FINAL SUMMARY
# ==========================================================================
SUMMARY_LOG="${PHASE_LOG_DIR}/summary.log"

{
echo "================================================================"
echo "[$(ts)] PIPELINE COMPLETE: ${TASK_A} <-> ${TASK_B}"
echo "================================================================"
echo ""
echo "  Total wall time: $(elapsed_since $PIPELINE_START)"
echo ""
echo "  Phase 0: ${TASK_A} ${#AVAIL_A[@]}/${NUM_CLUSTERS}, ${TASK_B} ${#AVAIL_B[@]}/${NUM_CLUSTERS}"
echo "  Phase 1: ${P1_DONE}/${PHASE1_EXPECTED} done in $(elapsed_since $PHASE1_START)"
echo "  Phase 2: ${P2_DONE}/${PHASE2_EXPECTED} done in $(elapsed_since $PHASE2_START)"
echo ""
echo "  Logs: ${PHASE_LOG_DIR}/"
echo ""
echo "  Next steps:"
echo "    bash scripts/transfer/check_status.sh"
echo "    python scripts/transfer/aggregate_demo_results.py \\"
echo "      --transfer_dir metamorph/output/transfer \\"
echo "      --loki_dir metamorph/output/loki \\"
echo "      --clusters \"${AVAIL_A[*]}\" \\"
echo "      --clusters_reverse \"${AVAIL_B[*]}\" \\"
echo "      --num_clusters ${NUM_CLUSTERS} --seed ${SEED}"
echo "================================================================"
} | tee "$SUMMARY_LOG"
