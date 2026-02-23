#!/usr/bin/env python3
"""
Build a cluster-task performance table from LOKI training results.

Reads Unimal-v0_results.json from each (cluster, task) output directory
and aggregates into CSV/JSON tables.

Usage:
    python tools/build_performance_table.py [--num_clusters 20] [--seed 3429] [--tail 50]

Output:
    results/cluster_task_table.json          - Full data with per-agent breakdown
    results/cluster_task_table_reward.csv    - Raw reward scores
    results/cluster_task_table_normalized.csv - Per-task normalized to [0,1]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


# Task name -> env_type used in output path
TASK_ENV_MAP = {
    "locomotion": "ft",
    "obstacle": "obstacle",
    "incline": "incline",
    "bump": "bump",
    "push_box_incline": "push_box_incline",
}


def get_results_path(base_dir, task, cluster_label, num_clusters, num_walkers,
                     drop_freq, num_drop, seed):
    env_type = TASK_ENV_MAP[task]
    return os.path.join(
        base_dir,
        env_type, "kmeans_cluster", str(num_clusters), str(cluster_label),
        f"walker{num_walkers}", f"freq{drop_freq}", f"drop{num_drop}",
        f"seed{seed}", "Unimal-v0_results.json"
    )


def extract_performance(results_path, tail_n=50):
    """Extract mean reward from the tail of training."""
    if not os.path.exists(results_path):
        return None, None

    with open(results_path) as f:
        data = json.load(f)

    env_data = data.get("__env__", {})
    reward_data = env_data.get("reward", {})
    reward_series = reward_data.get("reward", []) if isinstance(reward_data, dict) else []
    metric_series = env_data.get("metric", [])

    if not reward_series:
        return None, None

    tail_rewards = reward_series[-tail_n:]
    tail_metrics = metric_series[-tail_n:] if metric_series else []

    mean_reward = float(np.mean(tail_rewards))
    mean_metric = float(np.mean(tail_metrics)) if tail_metrics else None

    return mean_reward, mean_metric


def main():
    parser = argparse.ArgumentParser(description="Build cluster-task performance table")
    parser.add_argument("--base_dir", type=str,
                        default="./metamorph/output/loki",
                        help="Base output directory for LOKI runs")
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--num_walkers", type=int, default=20)
    parser.add_argument("--drop_freq", type=int, default=2)
    parser.add_argument("--num_drop", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3429)
    parser.add_argument("--tail", type=int, default=50,
                        help="Number of last log entries to average over")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["locomotion", "obstacle", "incline"],
                        help="Tasks to include in the table")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save output files")
    args = parser.parse_args()

    tasks = args.tasks
    os.makedirs(args.output_dir, exist_ok=True)

    # Build table
    table = {}
    for cluster_label in range(args.num_clusters):
        table[cluster_label] = {}
        for task in tasks:
            path = get_results_path(
                args.base_dir, task, cluster_label,
                args.num_clusters, args.num_walkers,
                args.drop_freq, args.num_drop, args.seed
            )
            reward, metric = extract_performance(path, args.tail)
            table[cluster_label][task] = {
                "mean_reward": reward,
                "mean_metric": metric,
                "results_path": path,
            }
            status = f"{reward:.2f}" if reward is not None else "MISSING"
            print(f"  Cluster {cluster_label:2d}, {task:20s}: reward={status}")

    # Save full JSON
    json_path = os.path.join(args.output_dir, "cluster_task_table.json")
    with open(json_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved {json_path}")

    # Save reward CSV
    csv_path = os.path.join(args.output_dir, "cluster_task_table_reward.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster"] + tasks)
        for cluster_label in range(args.num_clusters):
            row = [cluster_label]
            for task in tasks:
                val = table[cluster_label][task]["mean_reward"]
                row.append(f"{val:.4f}" if val is not None else "N/A")
            writer.writerow(row)
    print(f"Saved {csv_path}")

    # Normalized CSV (per-task min-max to [0,1])
    norm_path = os.path.join(args.output_dir, "cluster_task_table_normalized.csv")
    norm_table = {}
    for task in tasks:
        rewards = [
            table[c][task]["mean_reward"]
            for c in range(args.num_clusters)
            if table[c][task]["mean_reward"] is not None
        ]
        if not rewards:
            continue
        min_r, max_r = min(rewards), max(rewards)
        for cluster_label in range(args.num_clusters):
            if cluster_label not in norm_table:
                norm_table[cluster_label] = {}
            raw = table[cluster_label][task]["mean_reward"]
            if raw is not None and max_r > min_r:
                norm_table[cluster_label][task] = (raw - min_r) / (max_r - min_r)
            elif raw is not None:
                norm_table[cluster_label][task] = 1.0
            else:
                norm_table[cluster_label][task] = None

    with open(norm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster"] + tasks)
        for cluster_label in range(args.num_clusters):
            row = [cluster_label]
            for task in tasks:
                val = norm_table.get(cluster_label, {}).get(task)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            writer.writerow(row)
    print(f"Saved {norm_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("CLUSTER-TASK PERFORMANCE TABLE (mean reward, last {} log points)".format(args.tail))
    print(f"{'='*80}")
    header = f"{'Cluster':>8}" + "".join(f"{t:>18}" for t in tasks)
    print(header)
    print("-" * (8 + 18 * len(tasks)))
    found = 0
    missing = 0
    for cluster_label in range(args.num_clusters):
        row_str = f"{cluster_label:>8}"
        for task in tasks:
            val = table[cluster_label][task]["mean_reward"]
            if val is not None:
                row_str += f"{val:>18.2f}"
                found += 1
            else:
                row_str += f"{'N/A':>18}"
                missing += 1
        print(row_str)

    total = found + missing
    print(f"\nFound: {found}/{total} | Missing: {missing}/{total}")


if __name__ == "__main__":
    main()
