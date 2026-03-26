# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LOKI ("Convergent Functions, Divergent Forms") — a compute-efficient co-design framework for morphology + control policy, targeting MuJoCo-based UNIMAL creatures. Originally published at NeurIPS 2025 ([upstream repo](https://github.com/yeonsumia/loki)). This fork extends it with SLURM cluster support, LLM-guided cluster recommendation, morphology interpolation tools, and compatibility fixes for modern hardware (Ubuntu 24.04, RTX 5090, CUDA 12.8).

**Pipeline:** Random morphologies → VAE encoding (256-dim latent) → K-means clustering (40 clusters) → Cluster-specific PPO with elite replacement → Evaluate/fine-tune on downstream tasks.

## Installation

```bash
conda env create --file environment_cluster.yaml   # Python 3.12 (preferred for cluster)
conda activate loki
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
cd metamorph && pip install -e . && cd ../derl && pip install -e .
pip install -r requirements.txt
```

Requires MuJoCo 2.1.0 at `~/.mujoco/mujoco210` with `LD_LIBRARY_PATH` set. `environment.yaml` is the legacy env (Python 3.8).

## Key Commands

No test suites, linters, or CI pipelines. All training is launched via bash scripts in `scripts/`.

### Local Execution

```bash
# Data generation
bash scripts/evolve_init_xmls.sh <save_dir> <seed> <min_limbs> <max_limbs> <population_size>
bash scripts/data_2_wds.sh <data_dir>

# VAE training & clustering
bash scripts/train_vae.sh
bash scripts/make_clusters.sh <num_clusters>       # e.g. 40

# LOKI training (run per cluster)
bash scripts/train_loki.sh <num_walkers> <num_clusters> <cluster_idx> <seed>
bash scripts/train_loki_resume.sh <num_walkers> <num_clusters> <cluster_idx> <seed> <resume_iter>

# Elite extraction (manually define IDX_MAP in script first based on training results)
bash scripts/save_loki_top_agents.sh <num_clusters> <seed>

# Downstream task evaluation (supports parallelization via num_nodes)
bash scripts/eval_agents.sh <base_dir> <xml_dir> <task> <num_nodes> <node_idx>
bash scripts/get_top_eval_agents.sh <base_dir> <task> <top_k>

# Multi-agent fine-tuning on new tasks
bash scripts/finetune_loki.sh <num_walkers> <num_clusters> <cluster_idx> <seed> <task>
```

### SLURM Cluster Execution (this fork's additions)

```bash
# Data generation on cluster
bash scripts/gen_data_cluster.sh

# VAE training on cluster (with resource monitoring)
bash scripts/train_vae_cluster.sh

# Multi-task training across all clusters (two-phase scheduler)
bash scripts/train_loki_all_cluster_tasks.sh [num_gpus] [max_concurrent_per_gpu]
bash scripts/train_loki_single_task.sh <task>

# Cluster evaluation orchestrator
bash scripts/run_eval_cluster.sh

# Resume incomplete runs
bash scripts/schedule_resume_32.sh
bash scripts/schedule_remaining_28.sh
```

### Research Tools (this fork's additions)

```bash
# Build (cluster x task) performance table
python tools/build_performance_table.py --num_clusters 20 --seed 3429 --tail 50

# LLM-guided cluster recommendation for unseen tasks
export ANTHROPIC_API_KEY="..."
python tools/llm_cluster_recommender.py --new_task bump
python tools/llm_cluster_recommender_interactive.py   # interactive REPL mode

# Elite morphology interpolation in VAE latent space
MUJOCO_GL=egl python tools/interpolation_demo.py --new_task bump
MUJOCO_GL=egl python tools/midpoint_interpolation.py --new_task bump

# Visualization
MUJOCO_GL=egl python tools/visualize_best_performers.py
python tools/sample_cluster_morphologies.py --num_clusters 20 --num_samples 5
MUJOCO_GL=egl python tools/render_cluster_morphologies.py
python tools/visualize_cluster_grid.py --num_clusters 20 --num_samples 5
```

Python entry point for training: `python metamorph/tools/train_loki.py --cfg <config.yaml> --vae_path <path> [CONFIG OVERRIDES...]`

Tasks: `ft` (flat terrain/locomotion), `obstacle`, `many_obstacle`, `bump`, `incline`, `push_box_incline`, `manipulation_ball`, `exploration`, `patrol`.

## Architecture

### Three Packages (each with its own `setup.py`)

- **`metamorph/`** — Core training framework. YACS-based config (`metamorph/config.py`), PPO algorithm (`algos/ppo/`), multi-task environments (`envs/`), transformer-based actor-critic model.
- **`derl/`** — Morphology generation via evolutionary algorithm. Defines `SymmetricUnimal` morphology structure (tree with 4-10 limbs).
- **`vae/`** — Morphology VAE (256-dim latent) and K-means clustering. Trained on WebDataset of 50K morphologies.

### Core Algorithm (`metamorph/metamorph/algos/ppo/ppo_loki.py`)

The `LOKI` class extends PPO with morphology co-optimization:
1. Loads frozen VAE encoder and cluster morphologies
2. Maintains elite population (default 20 agents) per cluster
3. Training loop: collect rollouts → PPO update → periodically drop worst agents → mutate/resample from cluster via local search in VAE latent space
4. Elite replacement controlled by `LOKI.DROP_FREQ` and `LOKI.DROP_WARMUP`

### Configuration

YACS-based hierarchical config in `metamorph/metamorph/config.py`. Task-specific YAML overrides in `metamorph/configs/` (e.g., `ft.yaml`, `obstacle.yaml`, `bump.yaml`). CLI overrides use space-separated key-value pairs: `PPO.MAX_ITERS 500 LOKI.NUM_WALKER 20`.

Key sections: `PPO` (training hyperparams), `LOKI` (co-design params), `MODEL` (transformer architecture), `ENV` (task/environment), `TERRAIN`, `HFIELD`.

### Model (`metamorph/metamorph/algos/ppo/model.py`)

Transformer-based actor-critic (`TransformerModel`) with per-limb observation tokenization. Separate value and policy heads. Frozen VAE morphology latent injected as conditioning.

### Research Tools Pipeline (`tools/`)

The tools form a pipeline for the LLM-guided morphology selection research:
1. `build_performance_table.py` — Aggregate training results into (cluster, task) → reward matrix (JSON + CSV)
2. `llm_cluster_recommender.py` — Feed performance table to Claude API → predict best clusters for unseen tasks
3. `interpolation_demo.py` — Load recommended cluster's elite pool → sample/interpolate in VAE latent space (mean, reward-weighted mean with temperature)
4. `midpoint_interpolation.py` — Midpoint interpolation between top-2 unique elites (with MD5-based dedup)
5. `vec_to_morphology.py` — Decode VAE output back to valid MuJoCo XML (with repair functions for depth sequence and joints)

Outputs go to `results/` (recommendations, interpolation XMLs, visualizations).

### SLURM Scheduler Architecture (`scripts/train_loki_all_cluster_tasks.sh` and similar)

Two-phase scheduling pattern used across all cluster orchestration scripts:
- **Phase 1 (Staggered Launch):** One job per GPU with 5-min stagger to measure per-job resource usage
- **Phase 2 (Resource-Gated):** Round-robin GPU assignment, gated by: max concurrent limit, system RAM threshold (80 GB), and failure cooldown (OOM: 60s, fast-fail: 30s)
- GPU concurrency auto-detected by VRAM tier: A100-40GB → 5/GPU, A100-80GB → 10/GPU, H200 → 16/GPU
- Job completion detected by presence of `Unimal-v0_results.json` (only written after full training)
- Bash associative arrays track: `GPU_JOB_COUNT`, `PID_TO_GPU`, `PID_TO_JOB`, `PID_TO_START`

### Experiment Tracking

Weights & Biases (wandb). Defaults to online mode; falls back to offline if unavailable. On resume, uses deterministic run ID (`loki-${ENV_TYPE}-c${NUM_CLUSTERS}-idx${CLUSTER_LABEL}-w${NUM_WALKER}-s${RNG_SEED}`) to continue the same wandb run and preserve reward curves.

## Known Pitfalls & Fixes

These are issues already debugged and fixed in this fork — do not reintroduce:

- **Gymnasium API mismatch:** `metamorph/metamorph/algos/ppo/envs.py` monkeypatches Gymnasium wrappers to return old 4-tuple `(obs, reward, done, info)` format. New Gymnasium uses 5-tuple `(obs, reward, terminated, truncated, info)`. Don't remove the patch.
- **`OrderedDict` vs `dict`:** `metamorph/metamorph/envs/vec_env/utils.py` accepts both — newer Gymnasium returns plain `dict` for `spaces.Dict.spaces`. Don't tighten back to `OrderedDict`-only.
- **SubprocVecEnv OOM:** Use `DummyVecEnv` (in-process, `force_dummy=True`) instead of `SubprocVecEnv` for DLS evaluation sampling. Subprocess forking duplicates the full parent VM and triggers OOM on cluster.
- **DLS sampling batching:** Evaluate candidates in batches of `cfg.PPO.NUM_ENVS` rather than all at once, to control GPU memory. See `ppo_loki.py` eval methods.
- **Hash-based dedup:** Use MD5 hashing for morphology deduplication (`derl/derl/utils/similarity.py`) instead of O(N^2) all-pairs comparison, which causes OOM at scale.
- **wandb resume logging:** Deterministic `WANDB_RUN_ID` + elapsed time persistence in `elapsed_time.json` ensures reward curves and runtime stats survive crashes. Don't log `save_distance()` at iter 0 on resume (already logged in previous run).
- **Data generation fork bomb:** Limit `multiprocessing.Pool` to ~48 processes (not 128) in `gen_data_cluster.sh` — `fork()` duplicates parent VM with PyTorch loaded.
- **MuJoCo rendering on headless servers:** Always set `MUJOCO_GL=egl` (or `osmesa` as fallback) for offscreen rendering in tools.

## Codebase Conventions

- Config overrides on CLI: space-separated YACS keys `LOKI.TRAIN True PPO.MAX_ITERS 1000`
- Morphologies are XML files following UNIMAL schema (template: `metamorph/envs/assets/unimal.xml`)
- Output dirs: `metamorph/output/loki/<task>/kmeans_cluster/<num_clusters>/`
- Results/analysis output: `results/` (performance tables, LLM recommendations, visualizations)
- SLURM logs: `log/slurm/`
- `IDX_MAP` in `save_loki_top_agents.sh` must be manually defined from training results before extracting elites
