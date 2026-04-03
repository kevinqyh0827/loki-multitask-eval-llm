#!/bin/bash
# Single fine-tune or scratch training run for transfer learning experiments.
#
# This is the core building block for all transfer experiments. It handles:
# - Setting up walker directory (XML copy + PKL conversion)
# - Configuring fine-tune vs scratch mode
# - Supporting different model architectures (Transformer vs MLP)
# - Proper output directory organization
#
# Usage:
#   bash scripts/run_finetune_single.sh <mode> <target_task> <budget> <seed> <morph_xml_dir> [source_ckpt] [model_type] [source_name]
#
# Arguments:
#   mode:          "finetune" | "scratch" | "scratch_mlp"
#   target_task:   Target task name (e.g., "incline", "bump", "many_obstacle")
#   budget:        Total state-action pairs (e.g., "1e7" for 10M steps)
#   seed:          Random seed (e.g., 3429)
#   morph_xml_dir: Directory containing source morphology XML files (e.g., xml_step/1218/)
#   source_ckpt:   Path to source checkpoint .pt file (required for finetune mode)
#   model_type:    "ActorCritic" (default) or "MLPActorCritic"
#   source_name:   Name of source task for output dir (e.g., "ft", "push_box_incline", "bump")
#
# Examples:
#   # Fine-tune from ft policy on incline with 10M budget
#   bash scripts/run_finetune_single.sh finetune incline 1e7 3429 \
#     ./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429/xml_step/1218 \
#     ./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429/Unimal-v0.pt \
#     ActorCritic ft
#
#   # Scratch Transformer training on incline with 10M budget
#   bash scripts/run_finetune_single.sh scratch incline 1e7 3429 \
#     ./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429/xml_step/1218
#
#   # Scratch MLP training on incline with 10M budget
#   bash scripts/run_finetune_single.sh scratch_mlp incline 1e7 3429 \
#     ./output/loki/ft/kmeans_cluster/20/18/walker20/freq2/drop2/seed3429/xml_step/1218 \
#     "" MLPActorCritic

set -e

MODE=$1
TARGET_TASK=$2
BUDGET=$3
SEED=$4
MORPH_XML_DIR=$5
SOURCE_CKPT=${6:-""}
MODEL_TYPE=${7:-"ActorCritic"}
SOURCE_NAME=${8:-""}

if [ -z "$MODE" ] || [ -z "$TARGET_TASK" ] || [ -z "$BUDGET" ] || [ -z "$SEED" ] || [ -z "$MORPH_XML_DIR" ]; then
    echo "Usage: bash scripts/run_finetune_single.sh <mode> <target_task> <budget> <seed> <morph_xml_dir> [source_ckpt] [model_type] [source_name]"
    exit 1
fi

# Determine output directory based on mode
case $MODE in
    finetune)
        if [ -z "$SOURCE_CKPT" ]; then
            echo "Error: source_ckpt is required for finetune mode"
            exit 1
        fi
        OUT_BASE="./output/transfer/finetune_from_${SOURCE_NAME}/${TARGET_TASK}"
        ;;
    scratch)
        OUT_BASE="./output/transfer/scratch_transformer/${TARGET_TASK}"
        ;;
    scratch_mlp)
        MODEL_TYPE="MLPActorCritic"
        OUT_BASE="./output/transfer/scratch_mlp/${TARGET_TASK}"
        ;;
    *)
        echo "Error: mode must be 'finetune', 'scratch', or 'scratch_mlp'"
        exit 1
        ;;
esac

OUT_DIR="${OUT_BASE}/budget_${BUDGET}/seed_${SEED}"

# Check if already completed
if [ -f "metamorph/${OUT_DIR}/Unimal-v0_results.json" ]; then
    echo "Already completed: ${OUT_DIR}"
    exit 0
fi

echo "============================================"
echo "Mode:        $MODE"
echo "Target:      $TARGET_TASK"
echo "Budget:      $BUDGET"
echo "Seed:        $SEED"
echo "Model:       $MODEL_TYPE"
echo "Source:      ${SOURCE_NAME:-N/A}"
echo "Output:      $OUT_DIR"
echo "============================================"

# Setup walker directory with both xml/ and xml_step/0/ structures:
# - xml/ is used by train_ppo.py to infer walker list
# - xml_step/0/ is used by task.py (via LOKI.TRAIN) to load XMLs
WALKER_PATH="${OUT_DIR}/walkers"
mkdir -p "metamorph/${WALKER_PATH}/xml"
mkdir -p "metamorph/${WALKER_PATH}/xml_step/0"

# Copy morphology XML(s) - only copy numbered agent XMLs, not tmp_ files
# Handle paths with or without metamorph/ prefix
if [ -d "${MORPH_XML_DIR}" ]; then
    SRC_XML_DIR="${MORPH_XML_DIR}"
elif [ -d "metamorph/${MORPH_XML_DIR}" ]; then
    SRC_XML_DIR="metamorph/${MORPH_XML_DIR}"
else
    echo "Error: XML directory not found: ${MORPH_XML_DIR}"
    exit 1
fi

for xml_file in "${SRC_XML_DIR}"/[0-9]*.xml; do
    if [ -f "$xml_file" ]; then
        cp "$xml_file" "metamorph/${WALKER_PATH}/xml/"
        cp "$xml_file" "metamorph/${WALKER_PATH}/xml_step/0/"
    fi
done

# Adjust source checkpoint path to be relative to metamorph/
if [ -n "$SOURCE_CKPT" ]; then
    if [ -f "metamorph/${SOURCE_CKPT}" ]; then
        SOURCE_CKPT_REL="${SOURCE_CKPT}"
    elif [ -f "${SOURCE_CKPT}" ]; then
        # Strip metamorph/ prefix if present
        SOURCE_CKPT_REL="${SOURCE_CKPT#metamorph/}"
    else
        echo "Error: Checkpoint not found: ${SOURCE_CKPT}"
        exit 1
    fi
fi

cd metamorph

# Determine config file and extra args for target task
if [ "$TARGET_TASK" = "many_obstacle" ]; then
    CFG_FILE="./configs/obstacle.yaml"
    TASK_ARGS="OBJECT.NUM_OBSTACLES 150 ENV_TYPE many_obstacle"
else
    CFG_FILE="./configs/${TARGET_TASK}.yaml"
    TASK_ARGS="ENV_TYPE ${TARGET_TASK}"
fi

# Verify XMLs were copied successfully
XML_COUNT=$(ls "metamorph/${WALKER_PATH}/xml/"*.xml 2>/dev/null | wc -l)
if [ "$XML_COUNT" -eq 0 ]; then
    echo "Error: No XML files found in metamorph/${WALKER_PATH}/xml/"
    echo "SRC_XML_DIR was: ${SRC_XML_DIR}"
    exit 1
fi
echo "Walker dir ready: ${XML_COUNT} XMLs"

# Build common config overrides
# Use DummyVecEnv to avoid SubprocVecEnv fork OOM crashes (see CLAUDE.md pitfalls)
COMMON_ARGS="LOKI.TRAIN True \
    OUT_DIR ${OUT_DIR} \
    ENV.WALKER_DIR ${WALKER_PATH} \
    PPO.MAX_STATE_ACTION_PAIRS ${BUDGET} \
    RNG_SEED ${SEED} \
    MODEL.ACTOR_CRITIC ${MODEL_TYPE} \
    VECENV.TYPE DummyVecEnv \
    ${TASK_ARGS}"

# Add fine-tune specific args
if [ "$MODE" = "finetune" ]; then
    FINETUNE_ARGS="PPO.CHECKPOINT_PATH ${SOURCE_CKPT_REL} \
                   MODEL.FINETUNE.FULL_MODEL True"
else
    FINETUNE_ARGS=""
fi

# Create log directory
LOG_DIR="../log/transfer/${MODE}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TARGET_TASK}_budget${BUDGET}_seed${SEED}.log"

# Launch training
echo "Starting training..."
MUJOCO_GL=egl PYTHONPATH=./ python tools/train_ppo.py \
    --cfg "$CFG_FILE" \
    $COMMON_ARGS \
    $FINETUNE_ARGS \
    > "$LOG_FILE" 2>&1

echo "Training completed. Log: $LOG_FILE"
echo "Results: metamorph/${OUT_DIR}/Unimal-v0_results.json"
