"""Zero-shot policy evaluation on a target task.

Loads a pre-trained policy checkpoint and evaluates it on a different task
environment without any additional training. Used for measuring policy
transfer quality.

Usage:
    cd metamorph
    PYTHONPATH=./ python tools/eval_zero_shot.py \
        --cfg ./configs/incline.yaml \
        --checkpoint ./output/loki/ft/.../Unimal-v0.pt \
        --walker_dir /path/to/walker/dir \
        --num_episodes 50 \
        --out_dir ./output/transfer/zero_shot/ft_to_incline

    The walker_dir must contain xml_step/0/{agent_name}.xml files.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from metamorph.config import cfg
from metamorph.algos.ppo.envs import make_vec_envs, set_ob_rms
from metamorph.algos.ppo.model import Agent
from metamorph.utils import sample as su


def set_cfg_options():
    """Configure derived config values."""
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS)
        // cfg.PPO.TIMESTEPS
        // cfg.PPO.NUM_ENVS
    )
    if cfg.ENV_NAME == "Unimal-v0":
        cfg.MODEL.MAX_JOINTS = 20
        cfg.MODEL.MAX_LIMBS = 12


def maybe_infer_walkers():
    """Infer walker list from walker directory if not specified."""
    if cfg.ENV_NAME != "Unimal-v0":
        return
    if len(cfg.ENV.WALKERS):
        return
    xml_dir = os.path.join(cfg.ENV.WALKER_DIR, "xml_step", "0")
    cfg.ENV.WALKERS = [
        xml_file.split(".")[0]
        for xml_file in os.listdir(xml_dir)
        if xml_file.endswith(".xml") and not xml_file.startswith("tmp_")
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot policy evaluation")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file for target task", required=True, type=str
    )
    parser.add_argument(
        "--checkpoint", help="Path to source policy checkpoint (.pt)", required=True, type=str
    )
    parser.add_argument(
        "--walker_dir",
        help="Path to walker directory (containing xml_step/0/ subdir with XMLs)",
        required=True, type=str
    )
    parser.add_argument(
        "--num_episodes", help="Number of evaluation episodes", default=50, type=int
    )
    parser.add_argument(
        "--out_dir", help="Output directory for results", default="./output/transfer/zero_shot", type=str
    )
    parser.add_argument(
        "--device", help="Device to use", default="cuda:0", type=str
    )
    parser.add_argument(
        "opts",
        help="See metamorph/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def evaluate_zero_shot(args):
    """Run zero-shot evaluation: load checkpoint, run episodes, report rewards."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config for target task
    cfg.merge_from_file(args.cfg_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.DEVICE = args.device
    cfg.ENV.WALKER_DIR = args.walker_dir
    # Use LOKI.TRAIN to tell task.py to look in {WALKER_DIR}/xml_step/0/ for XMLs.
    # We do NOT use LOKI.FINETUNE because it triggers import of unimal_original.py
    # which requires mujoco_py (old bindings, not available on this machine).
    cfg.LOKI.TRAIN = True

    set_cfg_options()
    maybe_infer_walkers()

    # Set OUT_DIR and create sampling.json for MultiEnvWrapper.
    # MultiEnvWrapper reads {OUT_DIR}/sampling.json to decide which walker
    # each env instance switches to on reset. For eval, use round-robin.
    cfg.OUT_DIR = args.out_dir
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    num_walkers = len(cfg.ENV.WALKERS)
    num_envs = min(cfg.PPO.NUM_ENVS, 8)
    if num_walkers > 1:
        episodes_per_walker = max(args.num_episodes // num_walkers + 1, 5)
        sampling_seq = list(range(num_walkers)) * episodes_per_walker * num_envs
        sampling_path = os.path.join(cfg.OUT_DIR, "sampling.json")
        with open(sampling_path, "w") as f:
            json.dump(sampling_seq, f)

    print(f"Target task config: {args.cfg_file}")
    print(f"Walkers: {cfg.ENV.WALKERS}")
    print(f"Checkpoint: {args.checkpoint}")

    su.set_seed(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    # Create target task environment (no reward normalization, not training)
    # Use DummyVecEnv (force_dummy) to avoid SubprocVecEnv fork OOM issues
    # IMPORTANT: Must set cfg.PPO.NUM_ENVS to match actual env count, because
    # the model uses it as batch_size in forward() for reshaping observations.
    cfg.PPO.NUM_ENVS = num_envs
    envs = make_vec_envs(training=False, norm_rew=False, num_env=num_envs, force_dummy=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ac, ob_rms = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ac.to(device)
    ac.eval()
    set_ob_rms(envs, ob_rms)

    agent = Agent(ac)

    # Run evaluation episodes
    num_episodes = args.num_episodes
    episode_rewards = []
    episode_lengths = []

    obs = envs.reset()
    print(f"Running {num_episodes} evaluation episodes...")

    while len(episode_rewards) < num_episodes:
        val, act, logp = agent.act(obs)
        obs, reward, done, infos = envs.step(act)

        for info in infos:
            if "episode" in info:
                episode_rewards.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])
                if len(episode_rewards) % 10 == 0:
                    print(f"  Completed {len(episode_rewards)}/{num_episodes} episodes, "
                          f"running mean reward: {np.mean(episode_rewards):.1f}")

    # Compute statistics
    episode_rewards = episode_rewards[:num_episodes]
    episode_lengths = episode_lengths[:num_episodes]

    results = {
        "checkpoint": args.checkpoint,
        "target_task": args.cfg_file,
        "walker_dir": args.walker_dir,
        "walkers": cfg.ENV.WALKERS,
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "per_episode_rewards": [float(r) for r in episode_rewards],
        "per_episode_lengths": [int(l) for l in episode_lengths],
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Zero-Shot Evaluation Results")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target task: {args.cfg_file}")
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward:   {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")
    print(f"Median reward: {results['median_reward']:.1f}")
    print(f"Min/Max:       {results['min_reward']:.1f} / {results['max_reward']:.1f}")
    print(f"Mean ep length: {results['mean_episode_length']:.0f}")
    print("=" * 60)

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    envs.close()
    return results


def main():
    args = parse_args()
    evaluate_zero_shot(args)


if __name__ == "__main__":
    main()
