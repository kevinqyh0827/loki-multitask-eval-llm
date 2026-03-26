#!/bin/bash
# ==================================================================
# Train a single fixed morphology (best agent from Cluster 12 ft)
# on 3 tasks at paper-matched step budgets.
#
# Paper: arxiv 2505.21665, Section 4.2
#   5M steps  -> patrol, incline, obstacle, push_box_incline
#   15M steps -> bump
#   20M steps -> manipulate_ball, exploration
#
# Here we run: ft (5M), bump (15M), push_box_incline (20M)
#
# Usage:
#   bash scripts/train_single_morphology.sh
# ==================================================================

set -euo pipefail

# --- Configuration ---
CLUSTER_ID=12
BEST_AGENT_ID=17
SEED=3429
NUM_CLUSTERS=20

EVAL_DATA_ROOT="results/50k_20cluster_3tasks_eval_result/eval_record_details"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Source agent files
SRC_XML="${PROJECT_ROOT}/${EVAL_DATA_ROOT}/ft/kmeans_cluster/${NUM_CLUSTERS}/${CLUSTER_ID}/walker20/freq2/drop2/seed${SEED}/xml_step/1218/${BEST_AGENT_ID}.xml"
SRC_PKL="${PROJECT_ROOT}/${EVAL_DATA_ROOT}/ft/kmeans_cluster/${NUM_CLUSTERS}/${CLUSTER_ID}/walker20/freq2/drop2/seed${SEED}/unimal_pkl_recons_sample/1218/${BEST_AGENT_ID}.pkl"

VAE_PATH="VAE_50k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200_20260209_215748"

# Output base directory
OUTPUT_BASE="${PROJECT_ROOT}/metamorph/output/single_morph"

# Tasks and step budgets (task:steps)
declare -a EXPERIMENTS=(
    "ft:5e6"
    "bump:15e6"
    "push_box_incline:20e6"
)

# --- Verify source files exist ---
if [ ! -f "$SRC_XML" ]; then
    echo "ERROR: Source XML not found: $SRC_XML"
    exit 1
fi
if [ ! -f "$SRC_PKL" ]; then
    echo "ERROR: Source PKL not found: $SRC_PKL"
    exit 1
fi

echo "=================================================================="
echo "  Single Morphology Training — Cluster ${CLUSTER_ID}, Agent ${BEST_AGENT_ID}"
echo "=================================================================="
echo "  Source XML: $SRC_XML"
echo "  Source PKL: $SRC_PKL"
echo ""

# --- Setup init directory with the single agent ---
# The LOKI trainer uses INIT_DIR/xml_step/<iter>/ and INIT_DIR/unimal_pkl_recons_sample/<iter>/
INIT_DIR="${OUTPUT_BASE}/init_agent"
INIT_XML_DIR="${INIT_DIR}/xml_step/0"
INIT_PKL_DIR="${INIT_DIR}/unimal_pkl_recons_sample/0"

mkdir -p "$INIT_XML_DIR" "$INIT_PKL_DIR"

# Copy as agent 0 (the trainer loads agents 0..NUM_WALKER-1)
cp "$SRC_XML" "${INIT_XML_DIR}/0.xml"
cp "$SRC_PKL" "${INIT_PKL_DIR}/0.pkl"
echo "  Init directory prepared: $INIT_DIR"
echo ""

# --- Log directory ---
LOG_DIR="${PROJECT_ROOT}/log/single_morph"
mkdir -p "$LOG_DIR"

# --- Run experiments sequentially ---
SUMMARY=""

for experiment in "${EXPERIMENTS[@]}"; do
    IFS=":" read -r TASK STEPS <<< "$experiment"

    echo "=================================================================="
    echo "  Task: ${TASK} | Steps: ${STEPS}"
    echo "=================================================================="

    OUT_DIR="output/single_morph/${TASK}/${STEPS}/cluster${CLUSTER_ID}_agent${BEST_AGENT_ID}"
    LOG_FILE="${LOG_DIR}/${TASK}_${STEPS}.log"

    START_TIME=$(date +%s)

    cd "${PROJECT_ROOT}/metamorph"

    WANDB_MODE=offline PYTHONPATH=./ python tools/train_loki.py \
        --cfg "./configs/${TASK}.yaml" \
        --vae_path "$VAE_PATH" \
        OUT_DIR "$OUT_DIR" \
        LOKI.TRAIN True \
        LOKI.NUM_WALKER 1 \
        LOKI.UPDATE_WALKER False \
        LOKI.DROP_FREQ 99999 \
        LOKI.DROP_WARMUP 99999 \
        LOKI.NUM_DROP_WALKER 0 \
        LOKI.INIT_DIR "${INIT_DIR}" \
        LOKI.RESUME_ITER 0 \
        LOKI.NUM_CLUSTERS "$NUM_CLUSTERS" \
        LOKI.CLUSTER_LABEL "$CLUSTER_ID" \
        LOKI.SAMPLE_SIZE 1 \
        PPO.MAX_STATE_ACTION_PAIRS "$STEPS" \
        LOG_PERIOD 10 \
        ENV.TYPE "$TASK" \
        RNG_SEED "$SEED" 2>&1 | tee "$LOG_FILE"

    cd "$PROJECT_ROOT"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$(echo "scale=1; $ELAPSED / 60" | bc)

    echo ""
    echo "  Task ${TASK} completed in ${ELAPSED_MIN} minutes (${ELAPSED}s)"
    echo ""

    SUMMARY="${SUMMARY}  ${TASK} (${STEPS} steps): ${ELAPSED_MIN} min\n"
done

echo "=================================================================="
echo "  All experiments complete!"
echo "=================================================================="
echo -e "  Summary:\n${SUMMARY}"
echo "=================================================================="
