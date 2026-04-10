#!/bin/bash
# =============================================================================
# Check completion status of transfer experiments
# =============================================================================
#
# Scans the output directories and reports what completed, what failed, and
# what's missing. Works for both zero-shot and fine-tuning experiments.
#
# Usage:
#   bash scripts/transfer/check_status.sh [transfer_output_dir]
#
# Expected directory structure:
#   metamorph/output/transfer_500k/  (or metamorph/output/transfer/ for old 20-cluster runs)
#     zero_shot/{pair}/c{C}/seed{S}/eval_results.json
#     finetune/{pair}/c{C}/steps_{B}/seed{S}/Unimal-v0_results.json
# =============================================================================

TRANSFER_DIR="${1:-metamorph/output/transfer_500k}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRANSFER_DIR="${PROJECT_ROOT}/${TRANSFER_DIR}"

echo "============================================================"
echo "  Transfer Experiment Status"
echo "  Dir: ${TRANSFER_DIR}"
echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# --- Phase 1: Zero-shot ---
echo ">>> PHASE 1: Zero-shot evaluations"
echo "    Expected output per run: eval_results.json"
echo "    Contains: mean_reward, std_reward, per_episode_rewards"
echo ""
ZS_DONE=0
ZS_FAIL=0
ZS_TOTAL=0

if [ -d "${TRANSFER_DIR}/zero_shot" ]; then
    # Handle both /c{C}/eval_results.json and /c{C}/seed{S}/eval_results.json
    while IFS= read -r result_file; do
        [ -z "$result_file" ] && continue
        ZS_TOTAL=$((ZS_TOTAL + 1))

        # Extract pair and cluster from path
        rel="${result_file#${TRANSFER_DIR}/zero_shot/}"
        pair=$(echo "$rel" | cut -d/ -f1)
        cluster=$(echo "$rel" | cut -d/ -f2)

        MEAN=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d[\"mean_reward\"]:.1f}')" 2>/dev/null || echo "ERR")
        STD=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d[\"std_reward\"]:.1f}')" 2>/dev/null || echo "?")
        NEPS=$(python3 -c "import json; d=json.load(open('$result_file')); print(d['num_episodes'])" 2>/dev/null || echo "?")

        echo "  [DONE] ${pair} ${cluster}: reward=${MEAN} +/- ${STD} (${NEPS} eps)"
        ZS_DONE=$((ZS_DONE + 1))
    done < <(find "${TRANSFER_DIR}/zero_shot" -name "eval_results.json" 2>/dev/null | sort)

    # Check for directories without results (started but not finished)
    while IFS= read -r walkers_dir; do
        [ -z "$walkers_dir" ] && continue
        out_dir=$(dirname "$walkers_dir")
        if [ ! -f "${out_dir}/eval_results.json" ]; then
            rel="${out_dir#${TRANSFER_DIR}/zero_shot/}"
            echo "  [FAIL] ${rel}: walkers prepared, no eval_results.json"
            ZS_FAIL=$((ZS_FAIL + 1))
        fi
    done < <(find "${TRANSFER_DIR}/zero_shot" -type d -name "walkers" 2>/dev/null | sort)
else
    echo "  (no zero_shot directory found)"
fi
echo ""
echo "  Phase 1 total: ${ZS_DONE} done, ${ZS_FAIL} failed/incomplete"
echo ""

# --- Phase 2: Fine-tuning ---
echo ">>> PHASE 2: Fine-tuning runs"
echo "    Expected output per run: Unimal-v0_results.json"
echo "    Contains: per-agent reward histories (dict of agent_id -> reward list)"
echo "    Also: Unimal-v0.pt (checkpoint), config.yaml"
echo "    WandB: project=LOKI-transfer, id=transfer-{src}-to-{tgt}-c{C}-steps{B}-s{S}"
echo ""
FT_DONE=0
FT_RUNNING=0
FT_FAIL=0
FT_TOTAL=0

declare -A FT_BY_PAIR   # pair -> "done/total"
declare -A FT_BY_BUDGET  # budget -> "done/total"

if [ -d "${TRANSFER_DIR}/finetune" ]; then
    # Find all seed directories (the leaf level)
    while IFS= read -r seed_dir; do
        [ -z "$seed_dir" ] && [ ! -d "$seed_dir" ] && continue
        FT_TOTAL=$((FT_TOTAL + 1))

        rel="${seed_dir#${TRANSFER_DIR}/finetune/}"
        pair=$(echo "$rel" | cut -d/ -f1)
        cluster=$(echo "$rel" | cut -d/ -f2)
        budget=$(echo "$rel" | cut -d/ -f3)
        seed=$(echo "$rel" | cut -d/ -f4)

        # Track by pair and budget
        FT_BY_PAIR[$pair]="${FT_BY_PAIR[$pair]:-0}"
        FT_BY_BUDGET[$budget]="${FT_BY_BUDGET[$budget]:-0}"

        if [ -f "${seed_dir}/Unimal-v0_results.json" ]; then
            echo "  [DONE] ${pair} ${cluster} ${budget} ${seed}"
            FT_DONE=$((FT_DONE + 1))
            FT_BY_PAIR[$pair]=$(( ${FT_BY_PAIR[$pair]} + 1 ))
            FT_BY_BUDGET[$budget]=$(( ${FT_BY_BUDGET[$budget]} + 1 ))
        elif [ -f "${seed_dir}/Unimal-v0.pt" ]; then
            echo "  [RUN?] ${pair} ${cluster} ${budget} ${seed}: has checkpoint, no final results"
            FT_RUNNING=$((FT_RUNNING + 1))
        elif [ -f "${seed_dir}/config.yaml" ]; then
            echo "  [FAIL] ${pair} ${cluster} ${budget} ${seed}: has config, no checkpoint"
            FT_FAIL=$((FT_FAIL + 1))
        else
            echo "  [----] ${pair} ${cluster} ${budget} ${seed}: empty directory"
            FT_FAIL=$((FT_FAIL + 1))
        fi
    done < <(find "${TRANSFER_DIR}/finetune" -type d -name "seed*" 2>/dev/null | sort)
else
    echo "  (no finetune directory found)"
fi

echo ""
echo "  Phase 2 total: ${FT_DONE} done, ${FT_RUNNING} possibly running, ${FT_FAIL} failed"

# Breakdown by pair
if [ ${#FT_BY_PAIR[@]} -gt 0 ]; then
    echo ""
    echo "  By direction:"
    for pair in $(echo "${!FT_BY_PAIR[@]}" | tr ' ' '\n' | sort); do
        echo "    ${pair}: ${FT_BY_PAIR[$pair]} done"
    done
fi

# Breakdown by budget
if [ ${#FT_BY_BUDGET[@]} -gt 0 ]; then
    echo ""
    echo "  By budget:"
    for budget in $(echo "${!FT_BY_BUDGET[@]}" | tr ' ' '\n' | sort); do
        echo "    ${budget}: ${FT_BY_BUDGET[$budget]} done"
    done
fi

echo ""

# --- Pipeline log check ---
echo ">>> Pipeline logs"
LATEST_LOG=$(ls -td "${PROJECT_ROOT}"/log/transfer/pipeline_* 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  Latest pipeline run: ${LATEST_LOG}"
    for f in "${LATEST_LOG}"/*.log; do
        [ -f "$f" ] || continue
        echo "    $(basename $f): $(wc -l < "$f") lines"
    done
else
    echo "  (no pipeline logs found)"
fi

echo ""
echo "============================================================"
echo "  OVERALL: zero-shot ${ZS_DONE} done | fine-tune ${FT_DONE} done, ${FT_RUNNING} running, ${FT_FAIL} failed"
echo "============================================================"
