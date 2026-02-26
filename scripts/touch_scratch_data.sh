#!/bin/bash
#SBATCH --job-name=touch-data
#SBATCH --partition=work1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# === Scratch Data Protection ===
# Prevents automatic deletion of files on /scratch by touching them every 14 days.
# Scratch policy: files are deleted if not accessed/modified/metadata-changed in 30 days.
# This job self-resubmits to run every 14 days (well within the 30-day window).
#
# Initial submit:
#   sbatch scripts/touch_scratch_data.sh
#
# Check status:
#   squeue -u yinhonq --name=touch-data
#
# Cancel the recurring job:
#   scancel --name=touch-data

SCRATCH_DIR="/scratch/yinhonq/loki_data"

echo "Touching all files in $SCRATCH_DIR at $(date)"

if [ -d "$SCRATCH_DIR" ]; then
    FILE_COUNT=$(find "$SCRATCH_DIR" -type f | wc -l)
    echo "Found $FILE_COUNT files to touch"
    find "$SCRATCH_DIR" -type f -exec touch {} +
    echo "Done touching files at $(date)"
else
    echo "WARNING: $SCRATCH_DIR does not exist!"
fi

# Self-resubmit in 14 days
sbatch --begin=now+14days scripts/touch_scratch_data.sh
echo "Resubmitted touch job for 14 days from now"
