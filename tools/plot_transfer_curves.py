"""Generate paper-quality figures for transfer learning experiments.

Produces:
  - Figure 1: Learning curves (reward vs steps) for all conditions
  - Figure 2: Bar chart comparing final reward at a fixed budget
  - Figure 3: Task similarity vs transfer quality scatter plot
  - Figure 4: Training wall-clock time comparison

Usage:
    python tools/plot_transfer_curves.py --base_dir metamorph/output/transfer
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Style configuration
CONDITION_STYLES = {
    "finetune_from_ft": {
        "label": "Fine-tune from ft (LOKI)",
        "color": "#e74c3c",
        "linestyle": "-",
        "marker": "o",
        "similarity": 0.45,
    },
    "finetune_from_push_box_incline": {
        "label": "Fine-tune from push_box_incline",
        "color": "#3498db",
        "linestyle": "-",
        "marker": "s",
        "similarity": 0.52,
    },
    "finetune_from_bump": {
        "label": "Fine-tune from bump",
        "color": "#2ecc71",
        "linestyle": "-",
        "marker": "^",
        "similarity": 0.42,
    },
    "scratch_transformer": {
        "label": "Scratch (Transformer)",
        "color": "#9b59b6",
        "linestyle": "--",
        "marker": "D",
        "similarity": None,
    },
    "scratch_mlp": {
        "label": "Scratch (MLP)",
        "color": "#f39c12",
        "linestyle": "--",
        "marker": "v",
        "similarity": None,
    },
}


def load_results_json(path):
    """Load Unimal-v0_results.json and extract reward time series."""
    with open(path, "r") as f:
        data = json.load(f)

    env_rewards = data.get("__env__", {}).get("reward", {}).get("reward", [])
    if not env_rewards:
        for key, val in data.items():
            if key.startswith("__"):
                continue
            env_rewards = val.get("reward", {}).get("reward", [])
            if env_rewards:
                break
    return env_rewards


def collect_all_results(base_dir, target_task="incline"):
    """Collect results organized by condition -> budget -> seed -> rewards."""
    results = defaultdict(lambda: defaultdict(dict))

    for condition_name in CONDITION_STYLES:
        condition_dir = os.path.join(base_dir, condition_name, target_task)
        if not os.path.exists(condition_dir):
            continue

        for budget_dir in sorted(os.listdir(condition_dir)):
            if not budget_dir.startswith("budget_"):
                continue
            budget = budget_dir.replace("budget_", "")
            budget_path = os.path.join(condition_dir, budget_dir)

            for seed_dir in sorted(os.listdir(budget_path)):
                if not seed_dir.startswith("seed_"):
                    continue
                seed = seed_dir.replace("seed_", "")
                results_file = os.path.join(budget_path, seed_dir, "Unimal-v0_results.json")

                if os.path.exists(results_file):
                    rewards = load_results_json(results_file)
                    if rewards:
                        results[condition_name][budget][seed] = rewards

    return results


def get_final_reward(rewards, tail=50):
    """Get final reward as mean of last `tail` entries."""
    tail = min(tail, len(rewards))
    return float(np.mean(rewards[-tail:]))


def plot_budget_vs_reward(results, out_dir, target_task="incline"):
    """Figure 1: Bar chart of final reward at each budget for all conditions."""
    fig, ax = plt.subplots(figsize=(14, 6))

    conditions = [c for c in CONDITION_STYLES if c in results]
    budgets = sorted(
        set(b for c in conditions for b in results[c]),
        key=lambda x: float(x)
    )

    n_conditions = len(conditions)
    n_budgets = len(budgets)
    bar_width = 0.8 / n_conditions
    x = np.arange(n_budgets)

    for i, condition in enumerate(conditions):
        style = CONDITION_STYLES[condition]
        means = []
        stds = []

        for budget in budgets:
            if budget in results[condition]:
                seed_rewards = [
                    get_final_reward(rewards)
                    for rewards in results[condition][budget].values()
                ]
                means.append(np.mean(seed_rewards))
                stds.append(np.std(seed_rewards))
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_conditions / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, means, bar_width,
            yerr=stds, capsize=3,
            label=style["label"],
            color=style["color"],
            alpha=0.85,
        )

    ax.set_xlabel("Training Budget (state-action pairs)", fontsize=12)
    ax.set_ylabel("Final Reward", fontsize=12)
    ax.set_title(f"Transfer Learning: Final Reward by Budget — Target: {target_task}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([format_budget(b) for b in budgets], fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "budget_vs_reward.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_learning_curves(results, out_dir, target_task="incline", zero_shot_reward=None):
    """Figure 2: Learning curves (reward vs iteration) for each condition at each budget."""
    budgets = sorted(
        set(b for c in results for b in results[c]),
        key=lambda x: float(x)
    )

    # One subplot per budget
    n_budgets = len(budgets)
    cols = min(3, n_budgets)
    rows = (n_budgets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, budget in enumerate(budgets):
        ax = axes[idx // cols][idx % cols]

        for condition in CONDITION_STYLES:
            if condition not in results or budget not in results[condition]:
                continue

            style = CONDITION_STYLES[condition]
            seed_curves = list(results[condition][budget].values())

            if not seed_curves:
                continue

            # Align curves to same length
            min_len = min(len(c) for c in seed_curves)
            aligned = np.array([c[:min_len] for c in seed_curves])

            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)

            iters = np.arange(len(mean_curve))

            ax.plot(iters, mean_curve, label=style["label"],
                    color=style["color"], linestyle=style["linestyle"], linewidth=1.5)
            ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve,
                           color=style["color"], alpha=0.15)

        if zero_shot_reward is not None:
            ax.axhline(y=zero_shot_reward, color="gray", linestyle=":",
                      linewidth=1, label="Zero-shot" if idx == 0 else None)

        ax.set_title(f"Budget: {format_budget(budget)}", fontsize=11)
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.3)

    # Remove empty subplots
    for idx in range(n_budgets, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(handles)),
              fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Learning Curves — Target: {target_task}", fontsize=14, y=1.04)
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_similarity_vs_transfer(results, out_dir, target_task="incline", reference_budget="1e7"):
    """Figure 3: Scatter plot of task similarity vs transfer quality."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get scratch baseline reward at reference budget
    scratch_reward = None
    if "scratch_transformer" in results and reference_budget in results["scratch_transformer"]:
        seed_rewards = [
            get_final_reward(r) for r in results["scratch_transformer"][reference_budget].values()
        ]
        scratch_reward = np.mean(seed_rewards)

    finetune_conditions = [
        c for c in CONDITION_STYLES
        if c.startswith("finetune_") and c in results and CONDITION_STYLES[c]["similarity"] is not None
    ]

    for condition in finetune_conditions:
        style = CONDITION_STYLES[condition]
        if reference_budget not in results[condition]:
            continue

        seed_rewards = [
            get_final_reward(r) for r in results[condition][reference_budget].values()
        ]

        mean_reward = np.mean(seed_rewards)
        std_reward = np.std(seed_rewards)

        # Transfer improvement over scratch
        if scratch_reward and scratch_reward > 0:
            improvement = (mean_reward - scratch_reward) / scratch_reward * 100
            improvement_std = std_reward / scratch_reward * 100
        else:
            improvement = mean_reward
            improvement_std = std_reward

        ax.errorbar(
            style["similarity"], improvement,
            yerr=improvement_std,
            marker=style["marker"], color=style["color"],
            markersize=12, capsize=5, linewidth=2,
            label=style["label"],
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5,
              label="Scratch baseline")

    ax.set_xlabel("Task Similarity to Target", fontsize=12)
    ax.set_ylabel("Reward Improvement over Scratch (%)", fontsize=12)
    ax.set_title(f"Task Similarity vs Transfer Quality — Budget: {format_budget(reference_budget)}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "similarity_vs_transfer.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_fixed_budget_comparison(results, out_dir, target_task="incline",
                                 reference_budget="1e7", zero_shot_reward=None):
    """Figure 4: Bar chart comparing all conditions at a fixed budget."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = [c for c in CONDITION_STYLES if c in results and reference_budget in results[c]]

    labels = []
    means = []
    stds = []
    colors = []

    for condition in conditions:
        style = CONDITION_STYLES[condition]
        seed_rewards = [
            get_final_reward(r) for r in results[condition][reference_budget].values()
        ]
        labels.append(style["label"])
        means.append(np.mean(seed_rewards))
        stds.append(np.std(seed_rewards))
        colors.append(style["color"])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)

    if zero_shot_reward is not None:
        ax.axhline(y=zero_shot_reward, color="gray", linestyle=":",
                  linewidth=2, label=f"Zero-shot: {zero_shot_reward:.0f}")
        ax.legend(fontsize=10)

    ax.set_ylabel("Final Reward", fontsize=12)
    ax.set_title(f"Comparison at Budget {format_budget(reference_budget)} — Target: {target_task}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 10,
               f"{mean:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "fixed_budget_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def format_budget(budget_str):
    """Format budget string for display (e.g., '1e7' -> '10M')."""
    val = float(budget_str)
    if val >= 1e6:
        return f"{val/1e6:.0f}M"
    elif val >= 1e3:
        return f"{val/1e3:.0f}K"
    return str(int(val))


def main():
    parser = argparse.ArgumentParser(description="Plot transfer learning results")
    parser.add_argument("--base_dir", default="metamorph/output/transfer")
    parser.add_argument("--target_task", default="incline")
    parser.add_argument("--reference_budget", default="1e7",
                       help="Budget to use for fixed-budget comparisons")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.base_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Collect results
    results = collect_all_results(args.base_dir, args.target_task)

    if not results:
        print(f"No results found in {args.base_dir}")
        return

    print(f"Found results for conditions: {list(results.keys())}")

    # Load zero-shot results
    zero_shot_reward = None
    zs_file = os.path.join(args.base_dir, "zero_shot", args.target_task, "eval_results.json")
    # Also check with agent subdirs
    if not os.path.exists(zs_file):
        zs_dir = os.path.join(args.base_dir, "zero_shot", args.target_task)
        if os.path.exists(zs_dir):
            for subdir in os.listdir(zs_dir):
                candidate = os.path.join(zs_dir, subdir, "eval_results.json")
                if os.path.exists(candidate):
                    zs_file = candidate
                    break

    if os.path.exists(zs_file):
        with open(zs_file, "r") as f:
            zs_data = json.load(f)
        zero_shot_reward = zs_data.get("mean_reward")
        print(f"Zero-shot reward: {zero_shot_reward:.1f}")

    # Generate plots
    plot_budget_vs_reward(results, out_dir, args.target_task)
    plot_learning_curves(results, out_dir, args.target_task, zero_shot_reward)
    plot_similarity_vs_transfer(results, out_dir, args.target_task, args.reference_budget)
    plot_fixed_budget_comparison(results, out_dir, args.target_task,
                                args.reference_budget, zero_shot_reward)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
