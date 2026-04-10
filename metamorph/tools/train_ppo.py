import argparse
import hashlib
import os
import random
import sys
import time

import torch
import wandb

from metamorph.algos.ppo.ppo_original import PPO
from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_walkers()
    calculate_max_limbs_joints()


def calculate_max_limbs_joints():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    # Add extra 1 for max_joints; needed for adding edge padding
    cfg.MODEL.MAX_JOINTS = 20
    cfg.MODEL.MAX_LIMBS = 12


def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_walkers():
    if cfg.ENV_NAME != "Unimal-v0":
        return

    # Only infer the walkers if this option was not specified
    if len(cfg.ENV.WALKERS):
        return

    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(os.path.join(cfg.ENV.WALKER_DIR, "xml")) if not xml_file.startswith("tmp_")
    ]


def get_hparams():
    hparam_path = os.path.join(cfg.OUT_DIR, "hparam.json")
    # For local sweep return
    if not os.path.exists(hparam_path):
        return {}

    hparams = {}
    varying_args = fu.load_json(hparam_path)
    flatten_cfg = swu.flatten(cfg)

    for k in varying_args:
        hparams[k] = flatten_cfg[k]

    return hparams


def cleanup_tensorboard():
    tb_dir = os.path.join(cfg.OUT_DIR, "tensorboard")

    # Assume there is only one sub_dir and break when it's found
    for content in os.listdir(tb_dir):
        content = os.path.join(tb_dir, content)
        if os.path.isdir(content):
            break

    # Return if no dir found
    if not os.path.isdir(content):
        return

    # Move all the event files from sub_dir to tb_idr
    for event_file in os.listdir(content):
        src = os.path.join(content, event_file)
        dst = os.path.join(tb_dir, event_file)
        fu.move_file(src, dst)

    # Delete the sub_dir
    os.rmdir(content)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    su.set_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)
    PPOTrainer = PPO()
    PPOTrainer.train()
    hparams = get_hparams()
    PPOTrainer.save_rewards(hparams=hparams)
    PPOTrainer.save_model()
    # cleanup_tensorboard()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    # Set cfg options which are inferred
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Initialize wandb with deterministic run ID for resume support.
    # Derive a unique, stable run ID from OUT_DIR so that:
    #   - each condition/budget/seed combo gets its own run
    #   - crashed jobs resume the same run instead of creating duplicates
    wandb_kwargs = {}
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    if not wandb_run_id:
        # Generate deterministic 8-char ID from the output directory path
        wandb_run_id = hashlib.md5(cfg.OUT_DIR.encode()).hexdigest()[:8]
    wandb_kwargs["id"] = wandb_run_id
    wandb_kwargs["resume"] = "allow"

    # Stagger concurrent wandb.init() calls to avoid API rate limits on HPC.
    # With 8+ concurrent jobs, a 0-30s window causes overlapping inits that
    # contend for the WandB API and timeout at 60s. Use a wider window and
    # longer timeout to prevent cascading failures.
    wandb_stagger_max = int(os.environ.get("WANDB_STAGGER_MAX", "120"))
    stagger = random.uniform(0, wandb_stagger_max)
    print(f"[WANDB] Staggering init by {stagger:.0f}s (max={wandb_stagger_max}s)", flush=True)
    time.sleep(stagger)

    project = os.environ.get("WANDB_PROJECT", "LOKI-transfer")
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    print(f"[WANDB] Initializing mode={wandb_mode}, project={project}, "
          f"run_id={wandb_run_id}, name={cfg.OUT_DIR}", flush=True)

    wandb_init_timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "180"))
    wandb_settings = wandb.Settings(init_timeout=wandb_init_timeout)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            wandb.init(project=project, name=cfg.OUT_DIR, mode=wandb_mode,
                       settings=wandb_settings, **wandb_kwargs)
            break
        except Exception as e:
            if attempt < max_retries:
                wait = 15 * attempt + random.uniform(0, 10)
                print(f"[WANDB] Init failed (attempt {attempt}/{max_retries}): {e}. "
                      f"Retrying in {wait:.0f}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"[WANDB] Init failed after {max_retries} attempts, "
                      f"falling back to offline: {e}", flush=True)
                try:
                    offline_settings = wandb.Settings(init_timeout=wandb_init_timeout)
                    wandb.init(project=project, name=cfg.OUT_DIR, mode="offline",
                               settings=offline_settings, **wandb_kwargs)
                except Exception as e2:
                    print(f"[WANDB] Offline init also failed: {e2}. "
                          f"Disabling wandb entirely.", flush=True)
                    wandb.init(mode="disabled")
    print(f"[WANDB] Ready — mode={wandb.run.settings.mode}, "
          f"url={getattr(wandb.run, 'url', None) or 'offline/disabled'}", flush=True)

    # Save the config
    dump_cfg()
    ppo_train()


if __name__ == "__main__":
    main()
