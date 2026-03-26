#!/usr/bin/env python3
"""Analyze morphology replacement patterns across clusters and tasks.

Reads dropped_agents.json and Unimal-v0_results.json from eval_record_details
to summarize replacement frequency, cycling, temporal patterns, and
cross-task/cross-cluster correlations.

Usage:
    python tools/analyze_replacement_patterns.py
    python tools/analyze_replacement_patterns.py --output_dir results/replacement_analysis
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_DATA_DIR = os.path.join(
    _PROJECT_ROOT, "results", "50k_20cluster_3tasks_eval_result", "eval_record_details"
)
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "replacement_analysis")

TASKS = ["ft", "incline", "obstacle"]
NUM_CLUSTERS = 20
SEED = 3429


def build_data_path(data_dir, task, cluster):
    return os.path.join(
        data_dir, task, "kmeans_cluster", "20", str(cluster),
        "walker20", "freq2", "drop2", f"seed{SEED}",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dropped_agents(path):
    """Load dropped_agents.json → dict of {int_iter: (dropped_id, replacement_id)}."""
    fpath = os.path.join(path, "dropped_agents.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        raw = json.load(f)
    return {int(k): (v[0], v[1]) for k, v in raw.items()}


def load_final_rewards(path):
    """Load per-agent final rewards from Unimal-v0_results.json."""
    fpath = os.path.join(path, "Unimal-v0_results.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        data = json.load(f)
    rewards = {}
    for key, val in data.items():
        if key.startswith("_") or key == "fps":
            continue
        try:
            r_list = val["reward"]["reward"]
            rewards[key] = r_list[-1] if r_list else None
        except (KeyError, TypeError, IndexError):
            continue
    return rewards


# ---------------------------------------------------------------------------
# Per-(cluster, task) analysis
# ---------------------------------------------------------------------------

def analyze_single(dropped, rewards):
    """Compute stats for a single (cluster, task) pair."""
    if dropped is None:
        return None

    total_drops = len(dropped)
    iters = sorted(dropped.keys())

    dropped_ids = [v[0] for v in dropped.values()]
    replacement_ids = [v[1] for v in dropped.values()]

    drop_counter = Counter(dropped_ids)
    repl_counter = Counter(replacement_ids)

    unique_dropped = len(drop_counter)
    unique_replacements = len(repl_counter)

    most_dropped_id, most_dropped_count = drop_counter.most_common(1)[0]
    most_used_repl_id, most_used_repl_count = repl_counter.most_common(1)[0]

    # Cycling: same (dropped, replacement) pair appearing multiple times
    pair_counter = Counter((d, r) for d, r in zip(dropped_ids, replacement_ids))
    cycling_pairs = {
        f"{d}->{r}": count for (d, r), count in pair_counter.items() if count >= 3
    }

    # Temporal: split into thirds
    n = len(iters)
    third = n // 3
    early_iters = iters[:third] if third > 0 else iters
    mid_iters = iters[third:2*third] if third > 0 else []
    late_iters = iters[2*third:] if third > 0 else []

    early_drops = len(early_iters)
    mid_drops = len(mid_iters)
    late_drops = len(late_iters)

    # Rewards for most-dropped and most-used-replacement
    most_dropped_reward = rewards.get(most_dropped_id) if rewards else None
    most_used_repl_reward = rewards.get(most_used_repl_id) if rewards else None

    # All agent rewards (for context)
    best_agent = None
    best_reward = -float("inf")
    worst_agent = None
    worst_reward = float("inf")
    if rewards:
        for aid, r in rewards.items():
            if r is not None:
                if r > best_reward:
                    best_reward = r
                    best_agent = aid
                if r < worst_reward:
                    worst_reward = r
                    worst_agent = aid

    return {
        "total_drops": total_drops,
        "iteration_range": [iters[0], iters[-1]] if iters else None,
        "unique_dropped": unique_dropped,
        "unique_replacements": unique_replacements,
        "most_dropped": {
            "agent_id": most_dropped_id,
            "count": most_dropped_count,
            "final_reward": most_dropped_reward,
        },
        "most_used_replacement": {
            "agent_id": most_used_repl_id,
            "count": most_used_repl_count,
            "final_reward": most_used_repl_reward,
        },
        "drop_frequency": dict(drop_counter.most_common()),
        "replacement_frequency": dict(repl_counter.most_common()),
        "cycling_pairs": cycling_pairs,
        "temporal": {
            "early_drops": early_drops,
            "mid_drops": mid_drops,
            "late_drops": late_drops,
        },
        "best_agent": {"agent_id": best_agent, "reward": best_reward},
        "worst_agent": {"agent_id": worst_agent, "reward": worst_reward},
    }


# ---------------------------------------------------------------------------
# Cross-task / cross-cluster analysis
# ---------------------------------------------------------------------------

def cross_task_analysis(all_stats):
    """For each cluster, check if the same agents are weak across tasks."""
    results = {}
    for cluster in range(NUM_CLUSTERS):
        task_most_dropped = {}
        for task in TASKS:
            s = all_stats.get((cluster, task))
            if s:
                task_most_dropped[task] = s["most_dropped"]["agent_id"]
        # Check overlap
        ids = list(task_most_dropped.values())
        overlap = len(ids) - len(set(ids))
        results[cluster] = {
            "most_dropped_per_task": task_most_dropped,
            "shared_weak_agents": overlap > 0,
            "overlap_count": overlap,
        }
    return results


def cross_cluster_analysis(all_stats):
    """For each task, compare replacement activity across clusters."""
    results = {}
    for task in TASKS:
        cluster_drops = {}
        cluster_rewards = {}
        for cluster in range(NUM_CLUSTERS):
            s = all_stats.get((cluster, task))
            if s:
                cluster_drops[cluster] = s["total_drops"]
                cluster_rewards[cluster] = s["best_agent"]["reward"]
        results[task] = {
            "drops_per_cluster": cluster_drops,
            "best_reward_per_cluster": cluster_rewards,
            "most_active_cluster": max(cluster_drops, key=cluster_drops.get) if cluster_drops else None,
            "least_active_cluster": min(cluster_drops, key=cluster_drops.get) if cluster_drops else None,
        }
        # Correlation between drops and best reward
        if len(cluster_drops) >= 3:
            cs = sorted(cluster_drops.keys())
            drops_arr = np.array([cluster_drops[c] for c in cs], dtype=float)
            rewards_arr = np.array([cluster_rewards.get(c, 0) for c in cs], dtype=float)
            if drops_arr.std() > 0 and rewards_arr.std() > 0:
                corr = np.corrcoef(drops_arr, rewards_arr)[0, 1]
                results[task]["drops_reward_correlation"] = float(corr)
    return results


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_heatmap(all_stats, output_dir):
    """Heatmap of total drops: clusters x tasks."""
    matrix = np.full((NUM_CLUSTERS, len(TASKS)), np.nan)
    for i, cluster in enumerate(range(NUM_CLUSTERS)):
        for j, task in enumerate(TASKS):
            s = all_stats.get((cluster, task))
            if s:
                matrix[i, j] = s["total_drops"]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, fontsize=11)
    ax.set_yticks(range(NUM_CLUSTERS))
    ax.set_yticklabels(range(NUM_CLUSTERS), fontsize=10)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title("Total Replacement Events per (Cluster, Task)", fontsize=13)

    # Annotate cells
    for i in range(NUM_CLUSTERS):
        for j in range(len(TASKS)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=8, color="black" if val < matrix[~np.isnan(matrix)].max() * 0.7 else "white")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=8, color="gray")

    fig.colorbar(im, ax=ax, label="Total drops")
    fig.tight_layout()
    path = os.path.join(output_dir, "heatmap_drops.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drop_frequency_distribution(all_stats, output_dir):
    """Histogram of per-agent drop frequencies across all (cluster, task) pairs."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 5), sharey=True)
    for j, task in enumerate(TASKS):
        all_freqs = []
        for cluster in range(NUM_CLUSTERS):
            s = all_stats.get((cluster, task))
            if s:
                all_freqs.extend(s["drop_frequency"].values())
        if all_freqs:
            axes[j].hist(all_freqs, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        axes[j].set_title(f"{task}", fontsize=12)
        axes[j].set_xlabel("Times dropped", fontsize=10)
        if j == 0:
            axes[j].set_ylabel("Count (agents)", fontsize=10)

    fig.suptitle("Distribution of Per-Agent Drop Frequencies (All Clusters)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "drop_frequency_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_temporal_patterns(all_stats, output_dir):
    """Bar chart of early/mid/late drops aggregated across clusters."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 5), sharey=False)
    for j, task in enumerate(TASKS):
        early, mid, late = 0, 0, 0
        count = 0
        for cluster in range(NUM_CLUSTERS):
            s = all_stats.get((cluster, task))
            if s:
                early += s["temporal"]["early_drops"]
                mid += s["temporal"]["mid_drops"]
                late += s["temporal"]["late_drops"]
                count += 1
        if count > 0:
            bars = [early / count, mid / count, late / count]
            axes[j].bar(["Early", "Mid", "Late"], bars,
                       color=["#4CAF50", "#FF9800", "#F44336"], edgecolor="white")
            axes[j].set_title(f"{task} (avg per cluster)", fontsize=12)
            axes[j].set_ylabel("Avg drops", fontsize=10)

    fig.suptitle("Temporal Distribution of Replacements", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "temporal_patterns.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cycling_summary(all_stats, output_dir):
    """Bar chart showing how many (cluster, task) pairs exhibit cycling."""
    cycling_counts = {}
    for task in TASKS:
        n_cycling = 0
        for cluster in range(NUM_CLUSTERS):
            s = all_stats.get((cluster, task))
            if s and s["cycling_pairs"]:
                n_cycling += 1
        cycling_counts[task] = n_cycling

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(TASKS, [cycling_counts[t] for t in TASKS],
           color=["#2196F3", "#4CAF50", "#FF9800"], edgecolor="white")
    ax.set_ylabel("Number of clusters with cycling (>=3 same pair)", fontsize=11)
    ax.set_title("Cycling Behavior Prevalence", fontsize=13)
    ax.set_ylim(0, NUM_CLUSTERS + 1)
    for i, task in enumerate(TASKS):
        ax.text(i, cycling_counts[task] + 0.3, str(cycling_counts[task]),
                ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "cycling_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_drops_vs_reward(all_stats, output_dir):
    """Scatter: total drops vs best reward per (cluster, task)."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 5))
    for j, task in enumerate(TASKS):
        drops, rewards, labels = [], [], []
        for cluster in range(NUM_CLUSTERS):
            s = all_stats.get((cluster, task))
            if s:
                drops.append(s["total_drops"])
                rewards.append(s["best_agent"]["reward"])
                labels.append(str(cluster))
        if drops:
            axes[j].scatter(drops, rewards, c="steelblue", s=60, edgecolors="white", zorder=3)
            for x, y, lbl in zip(drops, rewards, labels):
                axes[j].annotate(lbl, (x, y), textcoords="offset points",
                                xytext=(5, 5), fontsize=8, alpha=0.7)
        axes[j].set_title(f"{task}", fontsize=12)
        axes[j].set_xlabel("Total drops", fontsize=10)
        if j == 0:
            axes[j].set_ylabel("Best agent reward", fontsize=10)
        axes[j].grid(True, alpha=0.3)

    fig.suptitle("Replacement Activity vs. Best Performance", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "drops_vs_reward.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze morphology replacement patterns")
    parser.add_argument("--data_dir", type=str, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_clusters", type=int, default=NUM_CLUSTERS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and analyze all (cluster, task) pairs
    # -----------------------------------------------------------------------
    print("Loading data...")
    all_stats = {}
    loaded = 0
    for cluster in range(args.num_clusters):
        for task in TASKS:
            path = build_data_path(args.data_dir, task, cluster)
            dropped = load_dropped_agents(path)
            rewards = load_final_rewards(path)
            stats = analyze_single(dropped, rewards)
            if stats:
                all_stats[(cluster, task)] = stats
                loaded += 1
    print(f"  Loaded {loaded} (cluster, task) pairs")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n=== Summary Table: Total Drops per (Cluster, Task) ===")
    header = f"{'Cluster':>8} |"
    for t in TASKS:
        header += f" {t:>10} |"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for cluster in range(args.num_clusters):
        row = f"{cluster:>8} |"
        row_data = {"cluster": cluster}
        for task in TASKS:
            s = all_stats.get((cluster, task))
            if s:
                row += f" {s['total_drops']:>10} |"
                row_data[task] = s["total_drops"]
            else:
                row += f" {'N/A':>10} |"
                row_data[task] = None
        print(row)
        summary_rows.append(row_data)

    # -----------------------------------------------------------------------
    # Cross-task analysis
    # -----------------------------------------------------------------------
    print("\n=== Cross-Task Analysis: Shared Weak Agents ===")
    ct_analysis = cross_task_analysis(all_stats)
    shared_count = 0
    for cluster, info in sorted(ct_analysis.items()):
        if info["shared_weak_agents"]:
            shared_count += 1
            print(f"  Cluster {cluster}: SHARED weak agent across tasks — {info['most_dropped_per_task']}")
    print(f"  {shared_count}/{args.num_clusters} clusters share the same most-dropped agent across 2+ tasks")

    # -----------------------------------------------------------------------
    # Cross-cluster analysis
    # -----------------------------------------------------------------------
    print("\n=== Cross-Cluster Analysis: Drops vs. Performance ===")
    cc_analysis = cross_cluster_analysis(all_stats)
    for task, info in cc_analysis.items():
        corr = info.get("drops_reward_correlation", "N/A")
        corr_str = f"{corr:.3f}" if isinstance(corr, float) else corr
        print(f"  {task}: most active=cluster {info['most_active_cluster']}, "
              f"least active=cluster {info['least_active_cluster']}, "
              f"drops-reward correlation={corr_str}")

    # -----------------------------------------------------------------------
    # Key findings
    # -----------------------------------------------------------------------
    print("\n=== Key Findings ===")

    # 1. Overall replacement activity
    all_drops = [s["total_drops"] for s in all_stats.values()]
    print(f"  Replacement events: min={min(all_drops)}, max={max(all_drops)}, "
          f"mean={np.mean(all_drops):.0f}, median={np.median(all_drops):.0f}")

    # 2. Cycling prevalence
    cycling_count = sum(1 for s in all_stats.values() if s["cycling_pairs"])
    print(f"  Cycling behavior: {cycling_count}/{loaded} pairs show repeated same-pair replacements")

    # 3. Most common pattern
    all_cycling = {}
    for (c, t), s in all_stats.items():
        for pair, cnt in s["cycling_pairs"].items():
            all_cycling[f"c{c}_{t}_{pair}"] = cnt
    if all_cycling:
        top_cycle = max(all_cycling, key=all_cycling.get)
        print(f"  Most extreme cycling: {top_cycle} ({all_cycling[top_cycle]} times)")

    # 4. Temporal pattern
    early_total = sum(s["temporal"]["early_drops"] for s in all_stats.values())
    mid_total = sum(s["temporal"]["mid_drops"] for s in all_stats.values())
    late_total = sum(s["temporal"]["late_drops"] for s in all_stats.values())
    total = early_total + mid_total + late_total
    if total > 0:
        print(f"  Temporal split: early={early_total/total:.1%}, "
              f"mid={mid_total/total:.1%}, late={late_total/total:.1%}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print("\nSaving outputs...")

    # Summary CSV
    csv_path = os.path.join(args.output_dir, "summary_stats.csv")
    with open(csv_path, "w") as f:
        f.write("cluster," + ",".join(TASKS) + "\n")
        for row in summary_rows:
            vals = [str(row.get(t, "")) if row.get(t) is not None else "" for t in TASKS]
            f.write(f"{row['cluster']}," + ",".join(vals) + "\n")
    print(f"  Saved: {csv_path}")

    # Full analysis JSON
    json_data = {
        "per_cluster_task": {
            f"c{c}_{t}": s for (c, t), s in all_stats.items()
        },
        "cross_task_analysis": ct_analysis,
        "cross_cluster_analysis": {
            t: {k: v for k, v in info.items() if k != "drops_per_cluster"}
            for t, info in cc_analysis.items()
        },
        "summary": {
            "total_pairs_analyzed": loaded,
            "total_replacement_events": sum(all_drops),
            "mean_drops": float(np.mean(all_drops)),
            "median_drops": float(np.median(all_drops)),
            "cycling_prevalence": f"{cycling_count}/{loaded}",
            "temporal_split": {
                "early": early_total,
                "mid": mid_total,
                "late": late_total,
            },
        },
    }
    json_path = os.path.join(args.output_dir, "full_analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # Visualizations
    print("\nGenerating visualizations...")
    plot_heatmap(all_stats, args.output_dir)
    plot_drop_frequency_distribution(all_stats, args.output_dir)
    plot_temporal_patterns(all_stats, args.output_dir)
    plot_cycling_summary(all_stats, args.output_dir)
    plot_drops_vs_reward(all_stats, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
