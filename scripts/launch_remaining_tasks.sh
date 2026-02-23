#!/bin/bash
# Launch remaining tasks from the 3×3 matrix (clusters 0,2,18 × locomotion,obstacle,incline)
# after currently running processes finish.
#
# This script polls for completion of running jobs, then launches the next ones.
# It runs up to MAX_CONCURRENT training jobs simultaneously.
#
# Usage: bash scripts/launch_remaining_tasks.sh [max_concurrent]
# Default max_concurrent=2

MAX_CONCURRENT=${1:-2}
POLL_INTERVAL=120  # seconds between checks

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Define the full 3×3 matrix: cluster task
ALL_JOBS=(
    "0 locomotion"
    "0 obstacle"
    "0 incline"
    "2 locomotion"
    "2 obstacle"
    "2 incline"
    "18 locomotion"
    "18 obstacle"
    "18 incline"
)

# Map task to env type for checking output path
task_to_env() {
    case "$1" in
        locomotion) echo "ft" ;;
        *) echo "$1" ;;
    esac
}

is_completed() {
    local cluster=$1 task=$2
    local env_type=$(task_to_env "$task")
    local results="metamorph/output/loki/$env_type/kmeans_cluster/20/$cluster/walker20/freq2/drop2/seed3429/Unimal-v0_results.json"
    [ -f "$results" ]
}

is_running() {
    local cluster=$1 task=$2
    local env_type=$(task_to_env "$task")
    ps aux | grep "train_loki.py" | grep -v grep | grep "CLUSTER_LABEL $cluster " | grep "ENV.TYPE $env_type" | grep -q .
}

count_running() {
    # Count unique running jobs by (cluster, env_type) pairs, not subprocess workers
    local count=0
    for job in "${ALL_JOBS[@]}"; do
        local c=$(echo "$job" | awk '{print $1}')
        local t=$(echo "$job" | awk '{print $2}')
        if ! is_completed "$c" "$t" && is_running "$c" "$t"; then
            count=$((count + 1))
        fi
    done
    echo $count
}

launch_job() {
    local cluster=$1 task=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: cluster=$cluster task=$task"
    bash scripts/train_loki_task.sh 20 20 "$cluster" 3429 "$task"
}

echo "========================================"
echo "  LOKI Remaining Task Launcher"
echo "  Max concurrent jobs: $MAX_CONCURRENT"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo "========================================"

# Main loop
while true; do
    # Build list of pending jobs (not completed and not running)
    PENDING=()
    RUNNING_COUNT=$(count_running)

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Status check (running: $RUNNING_COUNT/$MAX_CONCURRENT)"

    for job in "${ALL_JOBS[@]}"; do
        cluster=$(echo "$job" | awk '{print $1}')
        task=$(echo "$job" | awk '{print $2}')

        if is_completed "$cluster" "$task"; then
            echo "  [DONE]    cluster=$cluster task=$task"
        elif is_running "$cluster" "$task"; then
            echo "  [RUNNING] cluster=$cluster task=$task"
        else
            echo "  [PENDING] cluster=$cluster task=$task"
            PENDING+=("$job")
        fi
    done

    # Check if all done
    if [ ${#PENDING[@]} -eq 0 ] && [ "$RUNNING_COUNT" -eq 0 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 9 jobs completed!"
        break
    fi

    # Launch jobs if we have capacity
    LAUNCHED=0
    for job in "${PENDING[@]}"; do
        RUNNING_COUNT=$(count_running)
        if [ "$RUNNING_COUNT" -ge "$MAX_CONCURRENT" ]; then
            echo "  At max capacity ($RUNNING_COUNT/$MAX_CONCURRENT), waiting..."
            break
        fi

        cluster=$(echo "$job" | awk '{print $1}')
        task=$(echo "$job" | awk '{print $2}')
        launch_job "$cluster" "$task"
        LAUNCHED=$((LAUNCHED + 1))
        sleep 5  # Brief pause between launches
    done

    if [ ${#PENDING[@]} -eq 0 ] && [ "$RUNNING_COUNT" -gt 0 ]; then
        echo "  No pending jobs, waiting for $RUNNING_COUNT running job(s) to finish..."
    fi

    sleep $POLL_INTERVAL
done

echo "Done! You can now run: python tools/build_performance_table.py"
