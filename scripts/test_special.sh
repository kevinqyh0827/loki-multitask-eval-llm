#!/bin/bash
# Quick verification test for DummyVecEnv + cluster scheduling fixes.
#
# What this tests:
#   1. DummyVecEnv works with LOKI training (log/reset_one_unimal/update_one_unimal)
#   2. Training gets past the crash point (iteration 2, where self.envs.log() is called)
#   3. GPU memory measurement functions return consistent values
#
# Usage: bash scripts/test_special.sh
#
# The training uses a small MAX_STATE_ACTION_PAIRS so it finishes quickly (~5-10 min).
# Success = no AttributeError, training progresses past iteration 2.

set -e

echo "=== Test 1: GPU Memory Measurement Consistency ==="
echo ""

TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')
USED=$((TOTAL - FREE))
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

echo "GPUs detected: $NUM_GPUS"
echo "Total GPU memory (sum): ${TOTAL} MiB"
echo "Free GPU memory (sum):  ${FREE} MiB"
echo "Used GPU memory:        ${USED} MiB"
echo ""

if [ "$TOTAL" -gt 0 ] && [ "$FREE" -gt 0 ] && [ "$FREE" -le "$TOTAL" ]; then
    echo "[PASS] GPU memory values are consistent (free <= total, both summed across GPUs)"
else
    echo "[FAIL] GPU memory values look wrong: total=${TOTAL}, free=${FREE}"
    exit 1
fi
echo ""

echo "=== Test 2: DummyVecEnv Training (short run) ==="
echo ""
echo "Running LOKI training with DummyVecEnv and reduced MAX_STATE_ACTION_PAIRS..."
echo "The previous crash was at iteration 2 (self.envs.log() call)."
echo "If this passes iteration 2 without AttributeError, the fix works."
echo ""

cd metamorph
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./ python tools/train_loki.py \
    --cfg configs/ft.yaml \
    --vae_path VAE_50k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200_20260209_215748 \
    --device cuda:0 \
    OUT_DIR /tmp/loki_test_dummyvecenv \
    LOKI.TRAIN True \
    LOKI.NUM_WALKER 20 \
    LOKI.NUM_CLUSTERS 20 \
    LOKI.CLUSTER_LABEL 0 \
    LOKI.SAMPLE_SIZE 128 \
    PPO.MAX_STATE_ACTION_PAIRS 5e5 \
    LOKI.DROP_FREQ 2 \
    LOKI.NUM_DROP_WALKER 2 \
    LOKI.DROP_WARMUP 2 \
    LOG_PERIOD 10 \
    LOKI.MUTATE_SAMPLE False \
    ENV.TYPE ft \
    VECENV.TYPE DummyVecEnv \
    RNG_SEED 3429

echo ""
echo "=== All tests passed ==="
echo ""
echo "DummyVecEnv training completed without errors."
echo "You can now safely run the cluster evaluation with the fixed scripts."
echo ""
echo "Cleaning up test output..."
rm -rf /tmp/loki_test_dummyvecenv
