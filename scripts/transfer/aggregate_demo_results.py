#!/usr/bin/env python3
"""Aggregate and display transfer experiment results for any task pair.

Reads zero-shot eval_results.json and fine-tuning Unimal-v0_results.json files,
prints a summary table showing transfer efficiency at each budget.

Usage:
    # Specific task pair (obstacle -> many_obstacle)
    python scripts/transfer/aggregate_demo_results.py \
        --task_a obstacle --task_b many_obstacle

    # ft <-> incline
    python scripts/transfer/aggregate_demo_results.py \
        --task_a ft --task_b incline

    # Custom paths
    python scripts/transfer/aggregate_demo_results.py \
        --task_a obstacle --task_b many_obstacle \
        --transfer_dir metamorph/output/transfer \
        --loki_dir metamorph/output/loki \
        --num_clusters 20 --seed 3429
"""

import argparse
import json
import os
import sys

import numpy as np


BUDGETS = ["2e6", "5e6", "1e7", "2e7", "5e7"]

# Map task names to LOKI output directory names
TASK_TO_DIR = {
    "ft": "ft", "locomotion": "ft",
    "obstacle": "obstacle", "many_obstacle": "many_obstacle",
    "bump": "bump", "incline": "incline",
    "push_box_incline": "push_box_incline",
    "manipulation_ball": "manipulation_ball",
    "exploration": "exploration", "patrol": "patrol",
}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_loki_baseline_reward(loki_dir, task, cluster, num_clusters, seed):
    """Best agent tail-50 reward from LOKI training on a task."""
    task_dir = TASK_TO_DIR.get(task, task)
    path = os.path.join(
        loki_dir, task_dir,
        f"kmeans_cluster/{num_clusters}/{cluster}/walker20/freq2/drop2/seed{seed}",
        "Unimal-v0_results.json",
    )
    data = load_json(path)
    if data is None:
        return None

    best = -float("inf")
    for aid, ad in data.items():
        if aid.startswith("_") or not isinstance(ad, dict):
            continue
        rewards = ad.get("reward", {}).get("reward", [])
        if rewards:
            tail = rewards[-50:] if len(rewards) >= 50 else rewards
            m = np.mean(tail)
            if m > best:
                best = m
    return best if best > -float("inf") else None


def get_zero_shot_reward(transfer_dir, src, tgt, cluster, seed):
    """Zero-shot mean reward. Checks both path formats."""
    # New format: c{cluster}/seed{seed}/eval_results.json
    path = os.path.join(
        transfer_dir, "zero_shot", f"{src}_to_{tgt}",
        f"c{cluster}", f"seed{seed}", "eval_results.json",
    )
    data = load_json(path)
    if data is None:
        # Old format: c{cluster}/eval_results.json
        path = os.path.join(
            transfer_dir, "zero_shot", f"{src}_to_{tgt}",
            f"c{cluster}", "eval_results.json",
        )
        data = load_json(path)
    if data is None:
        return None, None
    return data.get("mean_reward"), data.get("std_reward")


def get_finetune_reward(transfer_dir, src, tgt, cluster, budget, seed):
    """Fine-tuned best agent tail-50 reward."""
    path = os.path.join(
        transfer_dir, "finetune", f"{src}_to_{tgt}", f"c{cluster}",
        f"steps_{budget}", f"seed{seed}", "Unimal-v0_results.json",
    )
    data = load_json(path)
    if data is None:
        return None

    best = -float("inf")
    for aid, ad in data.items():
        if aid.startswith("_") or not isinstance(ad, dict):
            continue
        rewards = ad.get("reward", {}).get("reward", [])
        if rewards:
            tail = rewards[-50:] if len(rewards) >= 50 else rewards
            m = np.mean(tail)
            if m > best:
                best = m
    return best if best > -float("inf") else None


def print_direction(transfer_dir, loki_dir, src, tgt, clusters, num_clusters, seed):
    """Print results table for one transfer direction."""
    if not clusters:
        return 0, 0

    print(f"\n{'=' * 100}")
    print(f"  {src} -> {tgt}")
    print(f"{'=' * 100}\n")

    # Header
    cols = ["Cluster", f"Base({src[:6]})", f"Base({tgt[:6]})", "Zero-shot"]
    cols += [f"FT {b}" for b in BUDGETS]
    header = " | ".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))

    done_zs, done_ft = 0, 0

    for c in clusters:
        base_src = get_loki_baseline_reward(loki_dir, src, c, num_clusters, seed)
        base_tgt = get_loki_baseline_reward(loki_dir, tgt, c, num_clusters, seed)

        zs_mean, zs_std = get_zero_shot_reward(transfer_dir, src, tgt, c, seed)
        if zs_mean is not None:
            done_zs += 1

        ft_rewards = []
        for b in BUDGETS:
            r = get_finetune_reward(transfer_dir, src, tgt, c, b, seed)
            if r is not None:
                done_ft += 1
            ft_rewards.append(r)

        def fmt(v, ref=None):
            if v is None:
                return f"{'--':>14}"
            s = f"{v:>10.1f}"
            if ref is not None and ref > 0:
                pct = v / ref * 100
                return f"{s} ({pct:3.0f}%)"[:14].rjust(14)
            return f"{s:>14}"

        row = [
            f"{'C' + str(c):>14}",
            fmt(base_src),
            fmt(base_tgt),
            fmt(zs_mean, base_tgt),
        ]
        for r in ft_rewards:
            row.append(fmt(r, base_tgt))

        print(" | ".join(row))

    print(f"\n  Base({src[:6]})  = best agent tail-50 reward, LOKI trained on {src}")
    print(f"  Base({tgt[:6]})  = best agent tail-50 reward, LOKI trained on {tgt} (reference)")
    print(f"  Zero-shot   = {src} policy evaluated on {tgt}, no training")
    print(f"  FT {{budget}} = {src} policy fine-tuned on {tgt} for {{budget}} steps")
    print(f"  (%)         = percentage of Base({tgt[:6]}) achieved")

    return done_zs, done_ft


def main():
    parser = argparse.ArgumentParser(description="Aggregate transfer results")
    parser.add_argument("--task_a", required=True, help="First task (e.g. ft, obstacle)")
    parser.add_argument("--task_b", required=True, help="Second task (e.g. incline, many_obstacle)")
    parser.add_argument("--transfer_dir", default="metamorph/output/transfer_500k")
    parser.add_argument("--loki_dir", default="metamorph/output/loki_500k")
    parser.add_argument("--num_clusters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3429)
    args = parser.parse_args()

    # Auto-detect which clusters have results
    clusters_fwd = []
    clusters_rev = []
    for c in range(args.num_clusters):
        zs, _ = get_zero_shot_reward(args.transfer_dir, args.task_a, args.task_b, str(c), args.seed)
        ft = get_finetune_reward(args.transfer_dir, args.task_a, args.task_b, str(c), BUDGETS[0], args.seed)
        if zs is not None or ft is not None:
            clusters_fwd.append(str(c))

        zs, _ = get_zero_shot_reward(args.transfer_dir, args.task_b, args.task_a, str(c), args.seed)
        ft = get_finetune_reward(args.transfer_dir, args.task_b, args.task_a, str(c), BUDGETS[0], args.seed)
        if zs is not None or ft is not None:
            clusters_rev.append(str(c))

    # Also check LOKI baselines for clusters not yet in transfer results
    for c in range(args.num_clusters):
        sc = str(c)
        task_a_dir = TASK_TO_DIR.get(args.task_a, args.task_a)
        path = os.path.join(
            args.loki_dir, task_a_dir,
            f"kmeans_cluster/{args.num_clusters}/{c}/walker20/freq2/drop2/seed{args.seed}",
            "Unimal-v0_results.json",
        )
        if os.path.exists(path) and sc not in clusters_fwd:
            clusters_fwd.append(sc)

    total_zs, total_ft = 0, 0

    zs1, ft1 = print_direction(
        args.transfer_dir, args.loki_dir,
        args.task_a, args.task_b, clusters_fwd,
        args.num_clusters, args.seed,
    )
    total_zs += zs1
    total_ft += ft1

    zs2, ft2 = print_direction(
        args.transfer_dir, args.loki_dir,
        args.task_b, args.task_a, clusters_rev,
        args.num_clusters, args.seed,
    )
    total_zs += zs2
    total_ft += ft2

    expected_zs = len(clusters_fwd) + len(clusters_rev)
    expected_ft = (len(clusters_fwd) + len(clusters_rev)) * len(BUDGETS)

    print(f"\n{'=' * 60}")
    print(f"  Completion: zero-shot {total_zs}/{expected_zs}, fine-tune {total_ft}/{expected_ft}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
