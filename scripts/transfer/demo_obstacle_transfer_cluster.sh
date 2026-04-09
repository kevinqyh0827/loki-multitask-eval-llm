#!/bin/bash
# =============================================================================
# Demo Transfer Pipeline: obstacle <-> many_obstacle (HPC/SLURM version)
# =============================================================================
#
# Runs the full cross-task transfer experiment between "obstacle" and
# "many_obstacle" on the HPC cluster with GPU-aware parallel scheduling.
#
# Follows the same two-phase scheduling pattern as train_loki_all_cluster_tasks.sh:
#   Phase 1 (staggered): one job per GPU to measure resource usage
#   Phase 2 (resource-gated): round-robin with concurrency + RAM limits
#
# SLURM submission:
#   sbatch --gres=gpu:h200:2 --cpus-per-task=16 --mem=128G --time=1-00:00:00 \
#          --output=log/slurm/transfer-demo-%j.out \
#          --error=log/slurm/transfer-demo-%j.err \
#          scripts/transfer/demo_obstacle_transfer_cluster.sh
#
# Interactive:
#   bash scripts/transfer/demo_obstacle_transfer_cluster.sh [num_gpus] [max_per_gpu]
#
# What it runs (for N clusters):
#   Phase 1: Zero-shot evals      — 2N jobs  (obstacle->many_obstacle + reverse)
#   Phase 2: Fine-tune budget sweep — 2N * 5 budgets = 10N jobs
#   Total with 40 clusters: 40*2 + 40*2*5 = 480 jobs
#   Total with 20 clusters: 20*2 + 20*2*5 = 240 jobs
#
# Each zero-shot job takes ~3-5 min. Each fine-tune job takes 5-60 min
# depending on budget. With 2 H200 GPUs (~32 concurrent slots): ~4-6 hours.
# =============================================================================

#SBATCH --job-name=loki-transfer-demo
#SBATCH --partition=work1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=log/slurm/transfer-demo-%j.out
#SBATCH --error=log/slurm/transfer-demo-%j.err

set -e

# ==================== Configuration ====================
NUM_GPUS=${1:-0}               # 0 = auto-detect
MAX_CONCURRENT_OVERRIDE=${2:-0} # 0 = auto-detect from VRAM tier

NUM_CLUSTERS=20
SEED=3429
NUM_WALKER=20
DROP_FREQ=2
NUM_DROP=2

# Fine-tuning budgets
BUDGETS=("2e6" "5e6" "1e7" "2e7" "5e7")

# Zero-shot episodes
ZERO_SHOT_EPISODES=50

# Scheduling — Phase 1 (zero-shot) is lightweight (~3-5 GB RAM per job)
# while Phase 2 (fine-tuning) is heavy (~15-25 GB RAM per job due to
# 20 walkers x 32 envs + optimizer + buffers). Use separate limits.
STAGGER_TIME_ZS=10       # Stagger for zero-shot (fast)
STAGGER_TIME_FT=30       # Stagger for fine-tune (allow RAM to settle)
RAM_THRESHOLD=40000      # 40 GB free RAM minimum

# Phase-level log directory
PIPELINE_START=$(date +%s)
PHASE_LOG_DIR="log/transfer/pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PHASE_LOG_DIR"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
elapsed_since() { echo "$(( $(date +%s) - $1 ))s"; }

# ==================== Cluster Environment Setup ====================
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running as SLURM job $SLURM_JOB_ID on $(hostname)"
    module load cuda/12.3 2>/dev/null || true
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate loki 2>/dev/null || true
fi

export MUJOCO_GL=egl
export WANDB_PROJECT="LOKI-transfer"

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

# Phase 1 (zero-shot): lightweight inference, many concurrent
# Phase 2 (fine-tune): heavy training (20 walkers, optimizer, buffers), few concurrent
#
# Per-job RAM usage:
#   zero-shot:  ~3-5 GB RAM,  ~2-3 GB VRAM
#   fine-tune:  ~15-25 GB RAM, ~8-12 GB VRAM
detect_max_concurrent_per_gpu() {
    local gpu_mem=$(get_gpu_total_mem 0)
    if [ "$gpu_mem" -le 50000 ]; then
        MAX_CONCURRENT_PER_GPU_ZS=5     # A100-40GB: zero-shot
        MAX_CONCURRENT_PER_GPU_FT=2     # A100-40GB: fine-tune
    elif [ "$gpu_mem" -le 100000 ]; then
        MAX_CONCURRENT_PER_GPU_ZS=10    # A100-80GB: zero-shot
        MAX_CONCURRENT_PER_GPU_FT=3     # A100-80GB: fine-tune
    else
        MAX_CONCURRENT_PER_GPU_ZS=16    # H200: zero-shot
        MAX_CONCURRENT_PER_GPU_FT=4     # H200: fine-tune
    fi
}

MAX_CONCURRENT_PER_GPU_ZS=0
MAX_CONCURRENT_PER_GPU_FT=0
if [ "$MAX_CONCURRENT_OVERRIDE" -gt 0 ]; then
    MAX_CONCURRENT_PER_GPU_ZS=$MAX_CONCURRENT_OVERRIDE
    MAX_CONCURRENT_PER_GPU_FT=$(( MAX_CONCURRENT_OVERRIDE < 4 ? MAX_CONCURRENT_OVERRIDE : 4 ))
else
    detect_max_concurrent_per_gpu
fi

# Active limit — switched between phases
TOTAL_MAX=$((NUM_GPUS * MAX_CONCURRENT_PER_GPU_ZS))

echo "================================================================"
echo "  Transfer Demo: obstacle <-> many_obstacle (HPC)"
echo "================================================================"
echo "  GPUs: ${NUM_GPUS}"
echo "  Phase 1 (zero-shot): max ${MAX_CONCURRENT_PER_GPU_ZS}/GPU = $((NUM_GPUS * MAX_CONCURRENT_PER_GPU_ZS)) total"
echo "  Phase 2 (fine-tune): max ${MAX_CONCURRENT_PER_GPU_FT}/GPU = $((NUM_GPUS * MAX_CONCURRENT_PER_GPU_FT)) total"
echo "  Clusters: ${NUM_CLUSTERS}, Seed: ${SEED}"
echo "  Budgets: ${BUDGETS[*]}"
echo "================================================================"
echo ""

# ==================== Scheduling State ====================
NEXT_GPU=0
declare -A GPU_JOB_COUNT
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_JOB_COUNT[$gpu]=0
done

RUNNING_PIDS=()
TOTAL_LAUNCHED=0
TOTAL_SKIPPED=0

# ==================== Helpers ====================
get_task_dir() {
    case "$1" in
        locomotion|ft) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

has_source_ckpt() {
    local task=$1 cluster=$2
    local dir=$(get_task_dir "$task")
    local path="metamorph/output/loki/${dir}/kmeans_cluster/${NUM_CLUSTERS}/${cluster}/walker${NUM_WALKER}/freq${DROP_FREQ}/drop${NUM_DROP}/seed${SEED}/Unimal-v0.pt"
    [ -f "$path" ]
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
            echo "[WAIT] RAM: ${free_ram}M free < ${RAM_THRESHOLD}M threshold (${#RUNNING_PIDS[@]} running)"
        fi
        sleep 5
    done
}

launch() {
    # launch <mode> <src> <tgt> <cluster> [budget]
    local mode=$1 src=$2 tgt=$3 cluster=$4 budget=${5:-""}

    wait_for_slot

    local gpu=$((NEXT_GPU % NUM_GPUS))
    NEXT_GPU=$(( (NEXT_GPU + 1) % NUM_GPUS ))

    local desc="${src}->${tgt} C${cluster}"
    [ -n "$budget" ] && desc="${desc} ${budget}"

    echo "[LAUNCH] GPU ${gpu}: ${desc} ($(update_running; echo ${#RUNNING_PIDS[@]})/${TOTAL_MAX} slots used)"

    CUDA_VISIBLE_DEVICES=$gpu \
    bash scripts/transfer/run_single_transfer.sh \
        "$mode" "$src" "$tgt" "$cluster" "$SEED" "$NUM_CLUSTERS" \
        "$budget" "$ZERO_SHOT_EPISODES" &

    RUNNING_PIDS+=($!)
    GPU_JOB_COUNT[$gpu]=$(( ${GPU_JOB_COUNT[$gpu]} + 1 ))
    TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

    # Use different stagger times: zero-shot is fast, fine-tune needs more breathing room
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
# PHASE 0: Inventory — find available source checkpoints
# ==========================================================================
#
# Expected output (logged to $PHASE_LOG_DIR/phase0_inventory.log):
#   - List of cluster IDs with completed obstacle training
#   - List of cluster IDs with completed many_obstacle training
#   - Total counts
#
# Expected files checked:
#   metamorph/output/loki/{obstacle,many_obstacle}/kmeans_cluster/{N}/{C}/
#     walker20/freq2/drop2/seed3429/Unimal-v0.pt
# ==========================================================================

PHASE0_START=$(date +%s)
PHASE0_LOG="${PHASE_LOG_DIR}/phase0_inventory.log"

echo "================================================================" | tee "$PHASE0_LOG"
echo "[$(ts)] PHASE 0: Inventory — checking source checkpoints"       | tee -a "$PHASE0_LOG"
echo "================================================================" | tee -a "$PHASE0_LOG"
echo ""                                                                | tee -a "$PHASE0_LOG"

AVAIL_OBS=()
AVAIL_MANY=()

for C in $(seq 0 $((NUM_CLUSTERS - 1))); do
    if has_source_ckpt "obstacle" "$C"; then
        AVAIL_OBS+=($C)
        echo "  [OK] obstacle      C${C}" | tee -a "$PHASE0_LOG"
    fi
    if has_source_ckpt "many_obstacle" "$C"; then
        AVAIL_MANY+=($C)
        echo "  [OK] many_obstacle C${C}" | tee -a "$PHASE0_LOG"
    fi
done

echo ""                                                                                | tee -a "$PHASE0_LOG"
echo "  obstacle checkpoints:      ${#AVAIL_OBS[@]}/${NUM_CLUSTERS} clusters"          | tee -a "$PHASE0_LOG"
echo "  many_obstacle checkpoints: ${#AVAIL_MANY[@]}/${NUM_CLUSTERS} clusters"         | tee -a "$PHASE0_LOG"
echo "  obstacle clusters:      [${AVAIL_OBS[*]}]"                                    | tee -a "$PHASE0_LOG"
echo "  many_obstacle clusters: [${AVAIL_MANY[*]}]"                                   | tee -a "$PHASE0_LOG"

P0_EXPECTED_FWD=${#AVAIL_OBS[@]}
P0_EXPECTED_REV=${#AVAIL_MANY[@]}

echo ""                                                                                | tee -a "$PHASE0_LOG"
echo "  Jobs planned:"                                                                 | tee -a "$PHASE0_LOG"
echo "    Phase 1 (zero-shot):  $((P0_EXPECTED_FWD + P0_EXPECTED_REV))"                | tee -a "$PHASE0_LOG"
echo "    Phase 2 (fine-tune):  $(( (P0_EXPECTED_FWD + P0_EXPECTED_REV) * ${#BUDGETS[@]} ))" | tee -a "$PHASE0_LOG"
echo ""                                                                                | tee -a "$PHASE0_LOG"
echo "[$(ts)] Phase 0 done in $(elapsed_since $PHASE0_START)"                          | tee -a "$PHASE0_LOG"
echo ""

if [ ${#AVAIL_OBS[@]} -eq 0 ] && [ ${#AVAIL_MANY[@]} -eq 0 ]; then
    echo "ERROR: No source checkpoints found. Exiting." | tee -a "$PHASE0_LOG"
    exit 1
fi

# ==========================================================================
# PHASE 1: Zero-shot evaluation
# ==========================================================================
#
# What it does:
#   Loads each cluster's trained policy (checkpoint + 20 morphology XMLs)
#   and evaluates it on the OTHER task for 50 episodes, with no fine-tuning.
#
# Expected output per job:
#   metamorph/output/transfer/zero_shot/{src}_to_{tgt}/c{C}/seed{S}/
#     eval_results.json    <- contains mean_reward, std_reward, per-episode data
#
# How to check:
#   - eval_results.json exists and has a valid "mean_reward" field
#   - Phase log: $PHASE_LOG_DIR/phase1_zero_shot.log
#
# Expected total jobs: #obstacle_clusters + #many_obstacle_clusters
# Expected time: ~3-5 min per job, all run in parallel -> ~5-15 min wall time
# ==========================================================================

PHASE1_START=$(date +%s)
PHASE1_LOG="${PHASE_LOG_DIR}/phase1_zero_shot.log"
PHASE1_EXPECTED=$((P0_EXPECTED_FWD + P0_EXPECTED_REV))

echo "================================================================" | tee "$PHASE1_LOG"
echo "[$(ts)] PHASE 1: Zero-shot evaluation"                           | tee -a "$PHASE1_LOG"
echo "  Expected jobs: ${PHASE1_EXPECTED}"                              | tee -a "$PHASE1_LOG"
echo "  Output: metamorph/output/transfer/zero_shot/{pair}/c{C}/seed${SEED}/" | tee -a "$PHASE1_LOG"
echo "  Result file: eval_results.json"                                 | tee -a "$PHASE1_LOG"
echo "================================================================" | tee -a "$PHASE1_LOG"
echo ""                                                                 | tee -a "$PHASE1_LOG"

# obstacle -> many_obstacle
if [ ${#AVAIL_OBS[@]} -gt 0 ]; then
    echo "--- obstacle -> many_obstacle (${#AVAIL_OBS[@]} clusters) ---" | tee -a "$PHASE1_LOG"
    for C in "${AVAIL_OBS[@]}"; do
        echo "[$(ts)] [LAUNCH] zero_shot obstacle->many_obstacle C${C}" | tee -a "$PHASE1_LOG"
        launch zero_shot obstacle many_obstacle "$C"
    done
fi

# many_obstacle -> obstacle
if [ ${#AVAIL_MANY[@]} -gt 0 ]; then
    echo "--- many_obstacle -> obstacle (${#AVAIL_MANY[@]} clusters) ---" | tee -a "$PHASE1_LOG"
    for C in "${AVAIL_MANY[@]}"; do
        echo "[$(ts)] [LAUNCH] zero_shot many_obstacle->obstacle C${C}" | tee -a "$PHASE1_LOG"
        launch zero_shot many_obstacle obstacle "$C"
    done
fi

echo ""                                                                 | tee -a "$PHASE1_LOG"
echo "[$(ts)] Waiting for ${TOTAL_LAUNCHED} zero-shot jobs..."          | tee -a "$PHASE1_LOG"
wait

# --- Phase 1 completion check ---
P1_DONE=0
P1_FAIL=0
P1_RESULTS=""

for C in "${AVAIL_OBS[@]}"; do
    F="metamorph/output/transfer/zero_shot/obstacle_to_many_obstacle/c${C}/seed${SEED}/eval_results.json"
    if [ -f "$F" ]; then
        MEAN=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"mean_reward\"]:.1f}')" 2>/dev/null || echo "?")
        echo "  [DONE] obstacle->many_obstacle C${C}: mean_reward=${MEAN}" | tee -a "$PHASE1_LOG"
        P1_RESULTS="${P1_RESULTS}  obstacle->many_obstacle C${C}: ${MEAN}\n"
        P1_DONE=$((P1_DONE + 1))
    else
        echo "  [FAIL] obstacle->many_obstacle C${C}: missing eval_results.json" | tee -a "$PHASE1_LOG"
        P1_FAIL=$((P1_FAIL + 1))
    fi
done

for C in "${AVAIL_MANY[@]}"; do
    F="metamorph/output/transfer/zero_shot/many_obstacle_to_obstacle/c${C}/seed${SEED}/eval_results.json"
    if [ -f "$F" ]; then
        MEAN=$(python3 -c "import json; d=json.load(open('$F')); print(f'{d[\"mean_reward\"]:.1f}')" 2>/dev/null || echo "?")
        echo "  [DONE] many_obstacle->obstacle C${C}: mean_reward=${MEAN}" | tee -a "$PHASE1_LOG"
        P1_RESULTS="${P1_RESULTS}  many_obstacle->obstacle C${C}: ${MEAN}\n"
        P1_DONE=$((P1_DONE + 1))
    else
        echo "  [FAIL] many_obstacle->obstacle C${C}: missing eval_results.json" | tee -a "$PHASE1_LOG"
        P1_FAIL=$((P1_FAIL + 1))
    fi
done

echo ""                                                                                 | tee -a "$PHASE1_LOG"
echo "================================================================"                 | tee -a "$PHASE1_LOG"
echo "[$(ts)] PHASE 1 SUMMARY"                                                         | tee -a "$PHASE1_LOG"
echo "  Duration: $(elapsed_since $PHASE1_START)"                                       | tee -a "$PHASE1_LOG"
echo "  Completed: ${P1_DONE}/${PHASE1_EXPECTED} (${P1_FAIL} failed)"                   | tee -a "$PHASE1_LOG"
echo ""                                                                                 | tee -a "$PHASE1_LOG"
echo "  Zero-shot rewards:"                                                             | tee -a "$PHASE1_LOG"
echo -e "$P1_RESULTS"                                                                   | tee -a "$PHASE1_LOG"
echo "================================================================"                 | tee -a "$PHASE1_LOG"
echo ""

# Reset counters for Phase 2 and switch to lower concurrency
PHASE1_TOTAL=$TOTAL_LAUNCHED
TOTAL_LAUNCHED=0
RUNNING_PIDS=()
TOTAL_MAX=$((NUM_GPUS * MAX_CONCURRENT_PER_GPU_FT))
echo "[$(ts)] Switching to Phase 2 concurrency: ${MAX_CONCURRENT_PER_GPU_FT}/GPU = ${TOTAL_MAX} total" | tee -a "$PHASE1_LOG"

# ==========================================================================
# PHASE 2: Fine-tuning budget sweep
# ==========================================================================
#
# What it does:
#   For each cluster, loads the trained policy from the source task, then
#   fine-tunes it on the target task at multiple training budgets.
#   Uses LOKI.TRAIN=True + PPO.CHECKPOINT_PATH for weight initialization.
#
# Expected output per job:
#   metamorph/output/transfer/finetune/{src}_to_{tgt}/c{C}/steps_{B}/seed{S}/
#     Unimal-v0_results.json    <- per-agent reward history (completion marker)
#     Unimal-v0.pt              <- final checkpoint
#     config.yaml               <- full config dump
#
# WandB:
#   Project: LOKI-transfer
#   Run ID:  transfer-{src}-to-{tgt}-c{C}-steps{B}-s{S}
#
# How to check:
#   - Unimal-v0_results.json exists (only written after full training completes)
#   - Phase log: $PHASE_LOG_DIR/phase2_finetune.log
#   - Per-job training log: log/transfer/finetune/{pair}/c{C}_steps{B}_seed{S}.log
#
# Expected total jobs: (#obstacle + #many_obstacle) * #budgets
# Expected time per job: 2e6 ~3min, 5e6 ~7min, 1e7 ~15min, 2e7 ~30min, 5e7 ~60min
# ==========================================================================

PHASE2_START=$(date +%s)
PHASE2_LOG="${PHASE_LOG_DIR}/phase2_finetune.log"
PHASE2_EXPECTED=$(( (P0_EXPECTED_FWD + P0_EXPECTED_REV) * ${#BUDGETS[@]} ))

echo "================================================================" | tee "$PHASE2_LOG"
echo "[$(ts)] PHASE 2: Fine-tuning budget sweep"                       | tee -a "$PHASE2_LOG"
echo "  Expected jobs: ${PHASE2_EXPECTED}"                              | tee -a "$PHASE2_LOG"
echo "  Budgets: ${BUDGETS[*]}"                                        | tee -a "$PHASE2_LOG"
echo "  Output: metamorph/output/transfer/finetune/{pair}/c{C}/steps_{B}/seed${SEED}/" | tee -a "$PHASE2_LOG"
echo "  Result file: Unimal-v0_results.json"                            | tee -a "$PHASE2_LOG"
echo "  WandB project: LOKI-transfer"                                   | tee -a "$PHASE2_LOG"
echo "================================================================" | tee -a "$PHASE2_LOG"
echo ""                                                                 | tee -a "$PHASE2_LOG"

# obstacle -> many_obstacle
if [ ${#AVAIL_OBS[@]} -gt 0 ]; then
    echo "--- obstacle -> many_obstacle (${#AVAIL_OBS[@]} clusters x ${#BUDGETS[@]} budgets) ---" | tee -a "$PHASE2_LOG"
    for C in "${AVAIL_OBS[@]}"; do
        for B in "${BUDGETS[@]}"; do
            echo "[$(ts)] [LAUNCH] finetune obstacle->many_obstacle C${C} ${B}" | tee -a "$PHASE2_LOG"
            launch finetune obstacle many_obstacle "$C" "$B"
        done
    done
fi

# many_obstacle -> obstacle
if [ ${#AVAIL_MANY[@]} -gt 0 ]; then
    echo "--- many_obstacle -> obstacle (${#AVAIL_MANY[@]} clusters x ${#BUDGETS[@]} budgets) ---" | tee -a "$PHASE2_LOG"
    for C in "${AVAIL_MANY[@]}"; do
        for B in "${BUDGETS[@]}"; do
            echo "[$(ts)] [LAUNCH] finetune many_obstacle->obstacle C${C} ${B}" | tee -a "$PHASE2_LOG"
            launch finetune many_obstacle obstacle "$C" "$B"
        done
    done
fi

echo ""                                                                 | tee -a "$PHASE2_LOG"
echo "[$(ts)] Waiting for ${TOTAL_LAUNCHED} fine-tune jobs..."          | tee -a "$PHASE2_LOG"
wait

# --- Phase 2 completion check ---
P2_DONE=0
P2_FAIL=0

for DIRECTION in "obstacle_to_many_obstacle:${AVAIL_OBS[*]}" "many_obstacle_to_obstacle:${AVAIL_MANY[*]}"; do
    PAIR="${DIRECTION%%:*}"
    CLUSTERS_STR="${DIRECTION#*:}"
    for C in $CLUSTERS_STR; do
        for B in "${BUDGETS[@]}"; do
            F="metamorph/output/transfer/finetune/${PAIR}/c${C}/steps_${B}/seed${SEED}/Unimal-v0_results.json"
            if [ -f "$F" ]; then
                echo "  [DONE] ${PAIR} C${C} steps_${B}" | tee -a "$PHASE2_LOG"
                P2_DONE=$((P2_DONE + 1))
            else
                echo "  [FAIL] ${PAIR} C${C} steps_${B}" | tee -a "$PHASE2_LOG"
                P2_FAIL=$((P2_FAIL + 1))
            fi
        done
    done
done

echo ""                                                                                 | tee -a "$PHASE2_LOG"
echo "================================================================"                 | tee -a "$PHASE2_LOG"
echo "[$(ts)] PHASE 2 SUMMARY"                                                         | tee -a "$PHASE2_LOG"
echo "  Duration: $(elapsed_since $PHASE2_START)"                                       | tee -a "$PHASE2_LOG"
echo "  Completed: ${P2_DONE}/${PHASE2_EXPECTED} (${P2_FAIL} failed)"                   | tee -a "$PHASE2_LOG"
echo "================================================================"                 | tee -a "$PHASE2_LOG"
echo ""

# ==========================================================================
# FINAL SUMMARY
# ==========================================================================

PIPELINE_ELAPSED=$(elapsed_since $PIPELINE_START)
SUMMARY_LOG="${PHASE_LOG_DIR}/summary.log"

{
echo "================================================================"
echo "[$(ts)] PIPELINE COMPLETE"
echo "================================================================"
echo ""
echo "  Total wall time: ${PIPELINE_ELAPSED}"
echo ""
echo "  Phase 0 (inventory):"
echo "    obstacle clusters:      ${#AVAIL_OBS[@]}/${NUM_CLUSTERS}"
echo "    many_obstacle clusters: ${#AVAIL_MANY[@]}/${NUM_CLUSTERS}"
echo ""
echo "  Phase 1 (zero-shot):"
echo "    Duration: $(elapsed_since $PHASE1_START)"
echo "    Completed: ${P1_DONE}/${PHASE1_EXPECTED}"
echo "    Results:"
echo -e "$P1_RESULTS"
echo ""
echo "  Phase 2 (fine-tune):"
echo "    Duration: $(elapsed_since $PHASE2_START)"
echo "    Completed: ${P2_DONE}/${PHASE2_EXPECTED}"
echo ""
echo "  Logs:"
echo "    ${PHASE_LOG_DIR}/phase0_inventory.log"
echo "    ${PHASE_LOG_DIR}/phase1_zero_shot.log"
echo "    ${PHASE_LOG_DIR}/phase2_finetune.log"
echo "    log/transfer/finetune/{pair}/c{C}_steps{B}_seed{S}.log  (per-job)"
echo ""
echo "  Next steps:"
echo "    # Check detailed status"
echo "    bash scripts/transfer/check_status.sh"
echo ""
echo "    # Aggregate into table"
echo "    python scripts/transfer/aggregate_demo_results.py \\"
echo "      --transfer_dir metamorph/output/transfer \\"
echo "      --loki_dir metamorph/output/loki \\"
echo "      --clusters \"${AVAIL_OBS[*]}\" \\"
echo "      --clusters_reverse \"${AVAIL_MANY[*]}\" \\"
echo "      --num_clusters ${NUM_CLUSTERS} --seed ${SEED}"
echo "================================================================"
} | tee "$SUMMARY_LOG"
