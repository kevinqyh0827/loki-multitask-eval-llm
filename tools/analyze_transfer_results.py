"""Analyze transfer learning experiment results.

Reads Unimal-v0_results.json files from all experiment directories,
computes statistics, and generates summary tables.

Usage:
    python tools/analyze_transfer_results.py --base_dir metamorph/output/transfer
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


# Condition metadata
CONDITIONS = {
    "finetune_from_ft": {
        "label": "Fine-tune from ft (LOKI DLS)",
        "color": "#e74c3c",
        "similarity": 0.45,
        "source": "ft",
    },
    "finetune_from_push_box_incline": {
        "label": "Fine-tune from push_box_incline",
        "color": "#3498db",
        "similarity": 0.52,
        "source": "push_box_incline",
    },
    "finetune_from_bump": {
        "label": "Fine-tune from bump",
        "color": "#2ecc71",
        "similarity": 0.42,
        "source": "bump",
    },
    "scratch_transformer": {
        "label": "Scratch (Transformer)",
        "color": "#9b59b6",
        "similarity": None,
        "source": None,
    },
    "scratch_mlp": {
        "label": "Scratch (MLP)",
        "color": "#f39c12",
        "similarity": None,
        "source": None,
    },
}


def load_results(results_path):
    """Load Unimal-v0_results.json and extract reward series."""
    with open(results_path, "r") as f:
        data = json.load(f)

    # Results format: {agent_name: {reward: {reward: [list]}, ...}, __env__: {...}}
    # For single-agent runs, there's one agent + __env__ aggregate
    env_rewards = data.get("__env__", {}).get("reward", {}).get("reward", [])
    if not env_rewards:
        # Try extracting from first agent
        for key, val in data.items():
            if key.startswith("__"):
                continue
            env_rewards = val.get("reward", {}).get("reward", [])
            if env_rewards:
                break

    return env_rewards


def find_experiment_dirs(base_dir, target_task="incline"):
    """Find all experiment result directories organized by condition."""
    results = defaultdict(lambda: defaultdict(dict))

    for condition_name in CONDITIONS:
        condition_dir = os.path.join(base_dir, condition_name, target_task)
        if not os.path.exists(condition_dir):
            continue

        for budget_dir in os.listdir(condition_dir):
            if not budget_dir.startswith("budget_"):
                continue
            budget = budget_dir.replace("budget_", "")
            budget_path = os.path.join(condition_dir, budget_dir)

            for seed_dir in os.listdir(budget_path):
                if not seed_dir.startswith("seed_"):
                    continue
                seed = seed_dir.replace("seed_", "")
                results_file = os.path.join(budget_path, seed_dir, "Unimal-v0_results.json")

                if os.path.exists(results_file):
                    results[condition_name][(budget, seed)] = results_file

    return results


def compute_statistics(experiment_dirs):
    """Compute per-condition, per-budget statistics across seeds."""
    stats = defaultdict(dict)

    for condition, runs in experiment_dirs.items():
        budget_rewards = defaultdict(list)
        budget_curves = defaultdict(list)

        for (budget, seed), results_file in runs.items():
            rewards = load_results(results_file)
            if rewards:
                # Final reward: average of last 50 entries (or all if fewer)
                tail = min(50, len(rewards))
                final_reward = np.mean(rewards[-tail:])
                budget_rewards[budget].append(final_reward)
                budget_curves[budget].append(rewards)

        for budget in sorted(budget_rewards.keys(), key=lambda x: float(x)):
            rews = budget_rewards[budget]
            stats[condition][budget] = {
                "mean_reward": float(np.mean(rews)),
                "std_reward": float(np.std(rews)),
                "min_reward": float(np.min(rews)),
                "max_reward": float(np.max(rews)),
                "n_seeds": len(rews),
                "per_seed_rewards": rews,
            }

    return stats


def compute_speedup(stats, baseline_condition="scratch_transformer", threshold_pct=0.8):
    """Compute speedup: steps for fine-tune to reach X% of scratch baseline performance."""
    speedup = {}

    if baseline_condition not in stats:
        return speedup

    # Get the best scratch baseline performance (highest budget)
    baseline_budgets = sorted(stats[baseline_condition].keys(), key=lambda x: float(x))
    if not baseline_budgets:
        return speedup

    best_baseline = stats[baseline_condition][baseline_budgets[-1]]["mean_reward"]
    target_reward = best_baseline * threshold_pct

    for condition in stats:
        if condition == baseline_condition:
            continue

        budgets = sorted(stats[condition].keys(), key=lambda x: float(x))
        reached_budget = None

        for budget in budgets:
            if stats[condition][budget]["mean_reward"] >= target_reward:
                reached_budget = budget
                break

        speedup[condition] = {
            "target_reward": float(target_reward),
            "baseline_best": float(best_baseline),
            "threshold_pct": threshold_pct,
            "reached_at_budget": reached_budget,
            "baseline_budget": baseline_budgets[-1],
        }

    return speedup


def load_zero_shot_results(base_dir):
    """Load zero-shot evaluation results if available."""
    zs_dir = os.path.join(base_dir, "zero_shot")
    results = {}

    if os.path.exists(zs_dir):
        for task_dir in os.listdir(zs_dir):
            eval_file = os.path.join(zs_dir, task_dir, "eval_results.json")
            if os.path.exists(eval_file):
                with open(eval_file, "r") as f:
                    results[task_dir] = json.load(f)

    return results


def print_summary(stats, speedup, zero_shot, target_task="incline"):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print(f"TRANSFER LEARNING RESULTS — Target: {target_task}")
    print("=" * 80)

    # Zero-shot results
    if zero_shot:
        print("\n--- Zero-Shot Transfer ---")
        for name, data in zero_shot.items():
            print(f"  {name}: {data['mean_reward']:.1f} +/- {data['std_reward']:.1f}")

    # Per-condition results
    print("\n--- Budget Sweep Results ---")
    print(f"{'Condition':<35} {'Budget':>8} {'Mean Reward':>12} {'Std':>8} {'Seeds':>6}")
    print("-" * 75)

    for condition in CONDITIONS:
        if condition not in stats:
            continue
        label = CONDITIONS[condition]["label"]
        for budget in sorted(stats[condition].keys(), key=lambda x: float(x)):
            s = stats[condition][budget]
            print(f"  {label:<33} {budget:>8} {s['mean_reward']:>12.1f} {s['std_reward']:>8.1f} {s['n_seeds']:>6}")

    # Speedup analysis
    if speedup:
        print("\n--- Speedup Analysis ---")
        for condition, sp in speedup.items():
            label = CONDITIONS.get(condition, {}).get("label", condition)
            reached = sp["reached_at_budget"]
            if reached:
                print(f"  {label}: reaches {sp['threshold_pct']*100:.0f}% of scratch baseline "
                      f"({sp['target_reward']:.0f}) at budget {reached}")
            else:
                print(f"  {label}: did NOT reach {sp['threshold_pct']*100:.0f}% of scratch baseline "
                      f"({sp['target_reward']:.0f}) at any tested budget")

    # Similarity analysis
    print("\n--- Task Similarity vs Transfer Quality ---")
    print(f"{'Source Task':<25} {'Similarity':>10} {'Best Reward':>12}")
    print("-" * 50)
    for condition in ["finetune_from_push_box_incline", "finetune_from_ft", "finetune_from_bump"]:
        if condition not in stats:
            continue
        meta = CONDITIONS[condition]
        budgets = sorted(stats[condition].keys(), key=lambda x: float(x))
        if budgets:
            best = max(stats[condition][b]["mean_reward"] for b in budgets)
            print(f"  {meta['source']:<23} {meta['similarity']:>10.3f} {best:>12.1f}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze transfer learning results")
    parser.add_argument(
        "--base_dir", default="metamorph/output/transfer",
        help="Base directory for transfer experiment outputs"
    )
    parser.add_argument(
        "--target_task", default="incline",
        help="Target task to analyze"
    )
    parser.add_argument(
        "--threshold", default=0.8, type=float,
        help="Performance threshold for speedup analysis (fraction of baseline)"
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory for analysis results (default: base_dir/analysis)"
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.base_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Find and load results
    experiment_dirs = find_experiment_dirs(args.base_dir, args.target_task)

    if not experiment_dirs:
        print(f"No results found in {args.base_dir}")
        return

    print(f"Found results for {len(experiment_dirs)} conditions")

    # Compute statistics
    stats = compute_statistics(experiment_dirs)
    speedup = compute_speedup(stats, threshold_pct=args.threshold)
    zero_shot = load_zero_shot_results(args.base_dir)

    # Print summary
    print_summary(stats, speedup, zero_shot, args.target_task)

    # Save summary JSON
    summary = {
        "target_task": args.target_task,
        "conditions": {},
        "speedup": speedup,
        "zero_shot": zero_shot,
    }
    for condition in stats:
        summary["conditions"][condition] = {
            "label": CONDITIONS.get(condition, {}).get("label", condition),
            "similarity": CONDITIONS.get(condition, {}).get("similarity"),
            "budgets": stats[condition],
        }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
