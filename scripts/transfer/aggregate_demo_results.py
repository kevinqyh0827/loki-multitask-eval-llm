#!/usr/bin/env python3
"""Aggregate and display transfer experiment results.

Reads zero-shot eval_results.json and fine-tuning Unimal-v0_results.json files
produced by the demo pipeline, and prints a summary table.

Usage:
    python scripts/transfer/aggregate_demo_results.py \
        --transfer_dir metamorph/output/transfer \
        --loki_dir metamorph/output/loki \
        --clusters "0 2 18" \
        --clusters_reverse "" \
        --num_clusters 20 \
        --seed 3429
"""

import argparse
import json
import os
import sys

import numpy as np


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_loki_baseline_reward(loki_dir, task_dir, cluster, num_clusters, seed):
    """Get the best agent reward from completed LOKI training on a task."""
    results_path = os.path.join(
        loki_dir, task_dir,
        f"kmeans_cluster/{num_clusters}/{cluster}/walker20/freq2/drop2/seed{seed}",
        "Unimal-v0_results.json",
    )
    data = load_json(results_path)
    if data is None:
        return None

    best_reward = -float("inf")
    for agent_id, agent_data in data.items():
        if agent_id.startswith("_"):
            continue
        rewards = agent_data.get("reward", {}).get("reward", [])
        if rewards:
            # Use mean of last 50 episodes (stable tail reward)
            tail = rewards[-50:] if len(rewards) >= 50 else rewards
            agent_mean = np.mean(tail)
            if agent_mean > best_reward:
                best_reward = agent_mean
    return best_reward if best_reward > -float("inf") else None


def get_zero_shot_reward(transfer_dir, src, tgt, cluster):
    """Get zero-shot mean reward."""
    path = os.path.join(transfer_dir, "zero_shot", f"{src}_to_{tgt}", f"c{cluster}", "eval_results.json")
    data = load_json(path)
    if data is None:
        return None, None
    return data.get("mean_reward"), data.get("std_reward")


def get_finetune_reward(transfer_dir, src, tgt, cluster, budget, seed):
    """Get fine-tuned mean reward (tail-50 of best agent)."""
    path = os.path.join(
        transfer_dir, "finetune", f"{src}_to_{tgt}", f"c{cluster}",
        f"steps_{budget}", f"seed{seed}", "Unimal-v0_results.json",
    )
    data = load_json(path)
    if data is None:
        return None

    best_reward = -float("inf")
    for agent_id, agent_data in data.items():
        if agent_id.startswith("_"):
            continue
        rewards = agent_data.get("reward", {}).get("reward", [])
        if rewards:
            tail = rewards[-50:] if len(rewards) >= 50 else rewards
            agent_mean = np.mean(tail)
            if agent_mean > best_reward:
                best_reward = agent_mean
    return best_reward if best_reward > -float("inf") else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer_dir", required=True)
    parser.add_argument("--loki_dir", required=True)
    parser.add_argument("--clusters", default="", help="Space-separated cluster IDs for obstacle->many_obstacle")
    parser.add_argument("--clusters_reverse", default="", help="Space-separated cluster IDs for many_obstacle->obstacle")
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3429)
    args = parser.parse_args()

    budgets = ["2e6", "5e6", "1e7", "2e7", "5e7"]
    clusters_fwd = [c for c in args.clusters.split() if c]
    clusters_rev = [c for c in args.clusters_reverse.split() if c]

    # --- obstacle -> many_obstacle ---
    if clusters_fwd:
        print("=" * 90)
        print("  obstacle -> many_obstacle")
        print("=" * 90)
        print("")

        # Header
        cols = ["Cluster", "Baseline(obs)", "Baseline(many)", "Zero-shot"] + [f"FT {b}" for b in budgets]
        header = " | ".join(f"{c:>14}" for c in cols)
        print(header)
        print("-" * len(header))

        for c in clusters_fwd:
            # Baseline: LOKI trained directly on each task
            base_obs = get_loki_baseline_reward(args.loki_dir, "obstacle", c, args.num_clusters, args.seed)
            base_many = get_loki_baseline_reward(args.loki_dir, "many_obstacle", c, args.num_clusters, args.seed)

            # Zero-shot
            zs_mean, zs_std = get_zero_shot_reward(args.transfer_dir, "obstacle", "many_obstacle", c)

            # Fine-tuning
            ft_rewards = []
            for b in budgets:
                r = get_finetune_reward(args.transfer_dir, "obstacle", "many_obstacle", c, b, args.seed)
                ft_rewards.append(r)

            # Format row
            def fmt(v, ref=None):
                if v is None:
                    return f"{'--':>14}"
                s = f"{v:>10.1f}"
                if ref is not None and ref > 0:
                    pct = v / ref * 100
                    s += f" ({pct:3.0f}%)"
                    return f"{s:>14}"
                return f"{s:>14}"

            row = [
                f"{'C' + c:>14}",
                fmt(base_obs),
                fmt(base_many),
                fmt(zs_mean, base_many),
            ]
            for r in ft_rewards:
                row.append(fmt(r, base_many))

            print(" | ".join(row))

        print("")
        print("  Baseline(obs)  = best agent tail-50 reward, LOKI trained on obstacle")
        print("  Baseline(many) = best agent tail-50 reward, LOKI trained on many_obstacle")
        print("  Zero-shot      = obstacle policy evaluated on many_obstacle (no training)")
        print("  FT {budget}    = obstacle policy fine-tuned on many_obstacle for {budget} steps")
        print("  (%)            = percentage of Baseline(many) achieved")
        print("")

    # --- many_obstacle -> obstacle ---
    if clusters_rev:
        print("=" * 90)
        print("  many_obstacle -> obstacle")
        print("=" * 90)
        print("")

        cols = ["Cluster", "Baseline(many)", "Baseline(obs)", "Zero-shot"] + [f"FT {b}" for b in budgets]
        header = " | ".join(f"{c:>14}" for c in cols)
        print(header)
        print("-" * len(header))

        for c in clusters_rev:
            base_many = get_loki_baseline_reward(args.loki_dir, "many_obstacle", c, args.num_clusters, args.seed)
            base_obs = get_loki_baseline_reward(args.loki_dir, "obstacle", c, args.num_clusters, args.seed)

            zs_mean, zs_std = get_zero_shot_reward(args.transfer_dir, "many_obstacle", "obstacle", c)

            ft_rewards = []
            for b in budgets:
                r = get_finetune_reward(args.transfer_dir, "many_obstacle", "obstacle", c, b, args.seed)
                ft_rewards.append(r)

            def fmt(v, ref=None):
                if v is None:
                    return f"{'--':>14}"
                s = f"{v:>10.1f}"
                if ref is not None and ref > 0:
                    pct = v / ref * 100
                    s += f" ({pct:3.0f}%)"
                    return f"{s:>14}"
                return f"{s:>14}"

            row = [
                f"{'C' + c:>14}",
                fmt(base_many),
                fmt(base_obs),
                fmt(zs_mean, base_obs),
            ]
            for r in ft_rewards:
                row.append(fmt(r, base_obs))

            print(" | ".join(row))

        print("")

    # --- Summary ---
    total_zs = 0
    total_ft = 0
    for c in clusters_fwd:
        if get_zero_shot_reward(args.transfer_dir, "obstacle", "many_obstacle", c)[0] is not None:
            total_zs += 1
        for b in budgets:
            if get_finetune_reward(args.transfer_dir, "obstacle", "many_obstacle", c, b, args.seed) is not None:
                total_ft += 1
    for c in clusters_rev:
        if get_zero_shot_reward(args.transfer_dir, "many_obstacle", "obstacle", c)[0] is not None:
            total_zs += 1
        for b in budgets:
            if get_finetune_reward(args.transfer_dir, "many_obstacle", "obstacle", c, b, args.seed) is not None:
                total_ft += 1

    expected_zs = len(clusters_fwd) + len(clusters_rev)
    expected_ft = (len(clusters_fwd) + len(clusters_rev)) * len(budgets)

    print(f"  Completion: zero-shot {total_zs}/{expected_zs}, fine-tune {total_ft}/{expected_ft}")


if __name__ == "__main__":
    main()
