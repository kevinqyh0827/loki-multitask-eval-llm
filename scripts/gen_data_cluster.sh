#!/bin/bash
#SBATCH --job-name=loki-datagen
#SBATCH --partition=work1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=log/slurm/loki-datagen-%j.out
#SBATCH --error=log/slurm/loki-datagen-%j.err

# === LOKI 500K Morphology Data Generation ===
# CPU-only job: generates 500K unique morphology samples (XML + PKL + PT vectors)
# then packages them into a webdataset tar and copies to scratch.
#
# Submit:
#   sbatch scripts/gen_data_cluster.sh
#
# The script generates 2x candidates (1M) and deduplicates to 500K unique morphologies.
# Estimated output size: ~10 GB (webdataset.tar)
# No GPU required — purely CPU multiprocessing.
#
# Why 48 processes instead of 128:
#   multiprocessing.Pool uses fork(), which duplicates the parent process's virtual
#   memory for each child. With a large parent (PyTorch + NetworkX imports), 128 forks
#   can exceed the SLURM memory allocation and trigger "Cannot allocate memory" (ENOMEM).
#   48 processes is a safe balance between parallelism and memory usage.

NUM_PROCESSES=48  # Parallel workers for morphology generation

# Setup environment
module load cuda/12.3
source /home/yinhonq/miniconda3/etc/profile.d/conda.sh
conda activate loki

# Navigate to repo
cd /home/yinhonq/test_pipelines/loki-multitask-eval-llm

# Create log directories
mkdir -p log/slurm

# Print job info
echo "=== LOKI Data Generation Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "Start time: $(date)"
echo ""

echo "--- System RAM ---"
free -h
echo ""

# --- Background resource monitor ---
MONITOR_LOG="log/slurm/loki-datagen-${SLURM_JOB_ID}-monitor.log"
MONITOR_INTERVAL=60  # seconds between snapshots

TARGET_TOTAL=1000000  # 2x 500K candidates generated before deduplication

format_duration() {
    local secs=$1
    printf "%dh %dm %ds" $(( secs / 3600 )) $(( (secs % 3600) / 60 )) $(( secs % 60 ))
}

monitor_resources() {
    echo "=== Resource Monitor Started (interval: ${MONITOR_INTERVAL}s) ===" > "$MONITOR_LOG"
    local monitor_start=$(date +%s)
    while true; do
        {
            local now=$(date +%s)
            local elapsed=$(( now - monitor_start ))
            echo "--- $(date '+%Y-%m-%d %H:%M:%S') [elapsed: $(format_duration $elapsed)] ---"

            # CPU usage (top 5 consumers)
            echo "[CPU] Load avg: $(cat /proc/loadavg)"
            echo "[CPU] Top processes:"
            ps -eo pid,user,%cpu,%mem,rss,comm --sort=-%cpu | head -6
            echo ""

            # Memory usage
            echo "[RAM] $(free -h | grep Mem)"
            echo "[RAM] $(free -h | grep Swap)"
            echo ""

            # Disk usage for output directories
            echo "[DISK] Working dir: $(du -sh derl/webdataset/ 2>/dev/null || echo 'N/A')"
            local xml_count=$(find derl/webdataset/ft/xml/ -type f 2>/dev/null | wc -l)
            local pkl_count=$(find derl/webdataset/ft/unimal_init/ -type f 2>/dev/null | wc -l)
            local vec_count=$(find derl/webdataset/ft/unimal_init_vec/ -type f 2>/dev/null | wc -l)
            echo "[DISK] XML count: $xml_count / $TARGET_TOTAL"
            echo "[DISK] PKL count: $pkl_count"
            echo "[DISK] Vec count: $vec_count"
            echo ""

            # ETA estimation based on XML file generation progress
            # Step 1 generates XMLs; Step 2 (pkl_2_vec) is much faster
            if [ "$xml_count" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
                local rate=$(echo "$xml_count $elapsed" | awk '{printf "%.1f", $1 / $2}')
                local remaining=$(( TARGET_TOTAL - xml_count ))
                if [ "$remaining" -gt 0 ] && [ "$xml_count" -lt "$TARGET_TOTAL" ]; then
                    local eta_secs=$(echo "$remaining $xml_count $elapsed" | awk '{printf "%d", ($1 / $2) * $3}')
                    echo "[ETA] Progress: $xml_count / $TARGET_TOTAL ($(echo "$xml_count $TARGET_TOTAL" | awk '{printf "%.1f", $1/$2*100}')%)"
                    echo "[ETA] Rate: $rate files/sec"
                    echo "[ETA] Estimated time remaining (Step 1): $(format_duration $eta_secs)"
                    echo "[ETA] Estimated completion: $(date -d "+${eta_secs} seconds" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'N/A')"
                elif [ "$xml_count" -ge "$TARGET_TOTAL" ]; then
                    echo "[ETA] Step 1 COMPLETE ($xml_count files generated in $(format_duration $elapsed))"
                    echo "[ETA] Now in deduplication or Step 2 (pkl_2_vec)..."
                fi
            else
                echo "[ETA] Waiting for first files to appear..."
            fi
            echo ""

            # Active python processes
            echo "[PROCS] Python processes: $(pgrep -c python 2>/dev/null || echo 0)"
            echo ""
        } >> "$MONITOR_LOG"
        sleep "$MONITOR_INTERVAL"
    done
}

monitor_resources &
MONITOR_PID=$!
echo "Resource monitor started (PID: $MONITOR_PID, log: $MONITOR_LOG)"
echo ""

# Cleanup monitor on exit
cleanup_monitor() {
    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null
        echo "Resource monitor stopped."
    fi
}
trap cleanup_monitor EXIT

# Clean up any artifacts from a previous failed run
if [ -d "derl/webdataset/ft" ] && [ ! -f "derl/webdataset/ft/init_setup_done" ]; then
    echo "=== Cleaning up previous failed run artifacts ==="
    rm -rf derl/webdataset/ft/xml/* derl/webdataset/ft/unimal_init/* derl/webdataset/ft/unimal_init_vec/* 2>/dev/null
    echo "Cleaned up partial files from previous run."
    echo ""
fi

# Step 1: Generate 500K morphologies (CPU-only)
STEP1_START=$(date +%s)
echo "=== Step 1: Generating 500K morphology XMLs (${NUM_PROCESSES} processes) ==="
bash scripts/evolve_init_xmls.sh webdataset 3429 4 10 500000 $NUM_PROCESSES
STEP1_END=$(date +%s)
STEP1_ELAPSED=$(( STEP1_END - STEP1_START ))
echo "Step 1 done at: $(date) (elapsed: $(( STEP1_ELAPSED / 3600 ))h $(( (STEP1_ELAPSED % 3600) / 60 ))m $(( STEP1_ELAPSED % 60 ))s)"
echo ""

# Step 2: Convert to webdataset tar format
STEP2_START=$(date +%s)
echo "=== Step 2: Converting to WebDataset tar ==="
bash scripts/data_2_wds.sh webdataset
STEP2_END=$(date +%s)
STEP2_ELAPSED=$(( STEP2_END - STEP2_START ))
echo "Step 2 done at: $(date) (elapsed: $(( STEP2_ELAPSED / 3600 ))h $(( (STEP2_ELAPSED % 3600) / 60 ))m $(( STEP2_ELAPSED % 60 ))s)"
echo ""

# Step 3: Copy results to scratch for persistence
echo "=== Step 3: Copying to scratch ==="
mkdir -p /scratch/yinhonq/loki_data
cp webdataset.tar /scratch/yinhonq/loki_data/webdataset_500k.tar
echo "Copied webdataset.tar to /scratch/yinhonq/loki_data/webdataset_500k.tar"
echo ""

# Final verification and summary
JOB_END=$(date +%s)
JOB_START_EPOCH=$(date -d "$(scontrol show job $SLURM_JOB_ID 2>/dev/null | grep StartTime | awk -F= '{print $2}')" +%s 2>/dev/null || echo "$STEP1_START")
TOTAL_ELAPSED=$(( JOB_END - JOB_START_EPOCH ))

echo "=== Verification ==="
echo "webdataset.tar size: $(du -h webdataset.tar | cut -f1)"
echo "XML file count: $(find derl/webdataset/ft/xml/ -type f 2>/dev/null | wc -l)"
echo "PKL file count: $(find derl/webdataset/ft/unimal_init/ -type f 2>/dev/null | wc -l)"
echo "Vec file count: $(find derl/webdataset/ft/unimal_init_vec/ -type f 2>/dev/null | wc -l)"
echo "Scratch copy size: $(du -h /scratch/yinhonq/loki_data/webdataset_500k.tar 2>/dev/null | cut -f1)"
echo ""

echo "=== Timing Summary ==="
echo "Step 1 (morphology generation): $(( STEP1_ELAPSED / 3600 ))h $(( (STEP1_ELAPSED % 3600) / 60 ))m $(( STEP1_ELAPSED % 60 ))s"
echo "Step 2 (webdataset conversion): $(( STEP2_ELAPSED / 3600 ))h $(( (STEP2_ELAPSED % 3600) / 60 ))m $(( STEP2_ELAPSED % 60 ))s"
echo "Total job time: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo ""

echo "=== Peak Resource Usage ==="
echo "[RAM] $(free -h | grep Mem)"
echo "Monitor log: $MONITOR_LOG"
echo ""

echo "End time: $(date)"
