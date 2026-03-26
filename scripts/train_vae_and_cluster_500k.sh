#!/bin/bash
#SBATCH --job-name=loki-vae-500k
#SBATCH --partition=work1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=log/slurm/loki-vae-500k-%j.out
#SBATCH --error=log/slurm/loki-vae-500k-%j.err

# === Train VAE on 500K dataset + Build K-means clusters ===
#
# Step 1: Train morphology VAE (200 epochs) on the 500K webdataset
# Step 2: Build K-means clusters (default 40) from the trained VAE's latent space
#
# Prerequisites:
#   - webdataset.tar (500K, ~9.9 GB) must exist in repo root
#   - OR /scratch/yinhonq/loki_data/webdataset_500k.tar as fallback
#
# Submit:
#   sbatch scripts/train_vae_and_cluster_500k.sh               # default 40 clusters
#   sbatch scripts/train_vae_and_cluster_500k.sh 20             # 20 clusters

NUM_CLUSTERS=${1:-40}

# Setup environment
module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki

# Navigate to repo
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

# Create log directories
mkdir -p log/slurm log/train_vae

# Print job info
echo "=== LOKI VAE Training (500K) + Clustering (${NUM_CLUSTERS} clusters) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

echo "--- GPU Details ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

echo "--- System RAM ---"
free -h
echo ""

# Symlink webdataset from scratch if not already present locally
if [ ! -f webdataset.tar ]; then
    echo "Symlinking webdataset from scratch..."
    ln -sf /scratch/yinhonq/loki_data/webdataset_500k.tar webdataset.tar
fi
echo "webdataset.tar size: $(du -h webdataset.tar | cut -f1)"
echo ""

# ============================================================
# Step 1: Train VAE
# ============================================================

BATCH_SIZE=4096
H_DIM=32
D_DEPTH=32
LR=1e-4
WD=1e-5
NUM_LAYER=4
NUM_HEAD=4
FACTOR=8
NUM_EPOCHS=200
DATASET_SIZE=500k

LOG_FILE="log/train_vae/VAE_${DATASET_SIZE}_hdim${H_DIM}_depth${D_DEPTH}_LR_${LR}_WD_${WD}_L${NUM_LAYER}_H${NUM_HEAD}_F${FACTOR}_beta0.01_bsize${BATCH_SIZE}_epochs${NUM_EPOCHS}.log"

echo "=== Step 1: Training VAE on 500K dataset (${NUM_EPOCHS} epochs) ==="
echo "Log file: $LOG_FILE"
echo ""

STEP1_START=$(date +%s)

PYTHONPATH=. python vae/train.py \
    --gpu 0 \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --min_beta 1e-5 \
    --max_beta 1e-2 \
    --epochs $NUM_EPOCHS \
    --factor $FACTOR \
    --n_head $NUM_HEAD \
    --n_layer $NUM_LAYER \
    --d_depth $D_DEPTH \
    --h_dim $H_DIM \
    --wd $WD \
    --dataset_size $DATASET_SIZE \
    --wds >> "$LOG_FILE" 2>&1

STEP1_EXIT=$?
STEP1_END=$(date +%s)
STEP1_ELAPSED=$(( STEP1_END - STEP1_START ))

echo "Step 1 finished at: $(date)"
echo "Step 1 duration: $(( STEP1_ELAPSED / 3600 ))h $(( (STEP1_ELAPSED % 3600) / 60 ))m $(( STEP1_ELAPSED % 60 ))s"
echo "Step 1 exit code: $STEP1_EXIT"
echo ""

if [ "$STEP1_EXIT" -ne 0 ]; then
    echo "[ERROR] VAE training failed (exit code $STEP1_EXIT). Check log: $LOG_FILE"
    echo "Last 20 lines:"
    tail -20 "$LOG_FILE"
    exit 1
fi

# Find the checkpoint directory that was just created
# Pattern: VAE_500k_hdim32_depth32_LR_..._<timestamp>
VAE_CKPT=$(ls -dt vae/checkpoints/VAE_${DATASET_SIZE}_hdim${H_DIM}_depth${D_DEPTH}_* 2>/dev/null | head -1)
VAE_CKPT_NAME=$(basename "$VAE_CKPT")

if [ -z "$VAE_CKPT" ] || [ ! -f "$VAE_CKPT/model.pt" ]; then
    echo "[ERROR] Could not find VAE checkpoint after training."
    echo "Checkpoints dir contents:"
    ls -lt vae/checkpoints/ | head -10
    exit 1
fi

echo "VAE checkpoint: $VAE_CKPT_NAME"
echo "Model file: $(ls -lh "$VAE_CKPT/model.pt")"
echo ""

# ============================================================
# Step 2: Build K-means clusters
# ============================================================

echo "=== Step 2: Building ${NUM_CLUSTERS} K-means clusters from VAE latent space ==="
echo ""

STEP2_START=$(date +%s)

PYTHONPATH=./ python vae/latent_cluster.py \
    --gpu 0 \
    --ckpt_dir "$VAE_CKPT_NAME" \
    --n_clusters $NUM_CLUSTERS \
    --output_dir data_500k \
    --make_cluster 2>&1 | tee "log/train_vae/clustering_${DATASET_SIZE}_${NUM_CLUSTERS}clusters.log"

STEP2_EXIT=$?
STEP2_END=$(date +%s)
STEP2_ELAPSED=$(( STEP2_END - STEP2_START ))

echo ""
echo "Step 2 finished at: $(date)"
echo "Step 2 duration: $(( STEP2_ELAPSED / 3600 ))h $(( (STEP2_ELAPSED % 3600) / 60 ))m $(( STEP2_ELAPSED % 60 ))s"
echo "Step 2 exit code: $STEP2_EXIT"
echo ""

if [ "$STEP2_EXIT" -ne 0 ]; then
    echo "[ERROR] Clustering failed (exit code $STEP2_EXIT)."
    exit 1
fi

# ============================================================
# Verification
# ============================================================

echo "=== Verification ==="

echo "--- VAE Checkpoint ---"
ls -lh "$VAE_CKPT"/model.pt "$VAE_CKPT"/encoder.pt "$VAE_CKPT"/decoder.pt 2>/dev/null
echo ""

echo "--- Cluster tar files (data_500k/) ---"
CLUSTER_COUNT=$(ls data_500k/latent_cluster${NUM_CLUSTERS}_*.tar 2>/dev/null | wc -l)
echo "Cluster tar files: $CLUSTER_COUNT / $NUM_CLUSTERS"
ls -lh data_500k/latent_cluster${NUM_CLUSTERS}_*.tar 2>/dev/null | head -5
echo "..."
ls -lh data_500k/latent_cluster${NUM_CLUSTERS}_*.tar 2>/dev/null | tail -5
echo ""

echo "Total cluster data size: $(du -sh data_500k/ 2>/dev/null | cut -f1)"
echo ""

echo "--- Cluster metadata ---"
CLUSTER_META_DIR=$(ls -d vae/latent_cluster${NUM_CLUSTERS}_new_webdataset 2>/dev/null)
if [ -n "$CLUSTER_META_DIR" ]; then
    ls -lh "$CLUSTER_META_DIR"/*.json "$CLUSTER_META_DIR"/*.pt 2>/dev/null
fi
echo ""

# ============================================================
# Summary
# ============================================================

JOB_END=$(date +%s)
TOTAL_ELAPSED=$(( JOB_END - STEP1_START ))

echo "=== Summary ==="
echo "VAE checkpoint: $VAE_CKPT_NAME"
echo "Clusters: $NUM_CLUSTERS (${CLUSTER_COUNT} tar files in data_500k/)"
echo "Step 1 (VAE training):  $(( STEP1_ELAPSED / 3600 ))h $(( (STEP1_ELAPSED % 3600) / 60 ))m"
echo "Step 2 (Clustering):    $(( STEP2_ELAPSED / 3600 ))h $(( (STEP2_ELAPSED % 3600) / 60 ))m"
echo "Total:                  $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m"
echo ""
echo "End time: $(date)"
