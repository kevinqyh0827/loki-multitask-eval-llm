#!/usr/bin/env python3
"""Analyze diversity collapse in LOKI elite pools across all cluster-task pairs.

For each of the 57 cluster-task pairs (20 clusters x 3 tasks), loads the final
elite pool (20 agents) and measures:
1. Unique morphologies via MD5 hash of XML files
2. VAE latent space mean distance (from distance_vector.pt)
3. Temporal evolution of diversity over training iterations
"""

import os
import sys
import json
import hashlib
import csv
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Two data sources: eval_record_details (all 57 pairs) and metamorph/output (local 9 pairs)
EVAL_ROOT = os.path.join(PROJECT_ROOT, "results", "50k_20cluster_3tasks_eval_result", "eval_record_details")
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "metamorph", "output", "loki")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "diversity_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = ["ft", "incline", "obstacle"]
NUM_CLUSTERS = 20
NUM_AGENTS = 20
SEED = 3429
SUFFIX = f"walker{NUM_AGENTS}/freq2/drop2/seed{SEED}"
FINAL_ITER = 1218


def get_base_dir(task, cluster_idx):
    """Find the base directory for a cluster-task pair (prefer eval_record_details)."""
    eval_path = os.path.join(EVAL_ROOT, task, "kmeans_cluster", str(NUM_CLUSTERS),
                             str(cluster_idx), SUFFIX)
    if os.path.isdir(eval_path):
        return eval_path
    train_path = os.path.join(TRAIN_ROOT, task, "kmeans_cluster", str(NUM_CLUSTERS),
                              str(cluster_idx), SUFFIX)
    if os.path.isdir(train_path):
        return train_path
    return None


def md5_file(path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_unique_xml(base_dir):
    """Count unique morphologies in final elite pool via MD5 of XML files."""
    xml_dir = os.path.join(base_dir, "xml_step", str(FINAL_ITER))
    if not os.path.isdir(xml_dir):
        return None, None, None

    hashes = {}
    for i in range(NUM_AGENTS):
        xml_path = os.path.join(xml_dir, f"{i}.xml")
        if os.path.exists(xml_path):
            h = md5_file(xml_path)
            hashes[i] = h

    if not hashes:
        return None, None, None

    # Group agents by identical hash
    hash_to_agents = defaultdict(list)
    for agent_id, h in hashes.items():
        hash_to_agents[h].append(agent_id)

    num_unique = len(hash_to_agents)
    # Largest group size (most duplicated morphology)
    max_group = max(len(v) for v in hash_to_agents.values())
    return num_unique, max_group, dict(hash_to_agents)


def count_unique_pkl(base_dir):
    """Count unique morphologies via MD5 of PKL files."""
    pkl_dir = os.path.join(base_dir, "unimal_pkl_recons_sample", str(FINAL_ITER))
    if not os.path.isdir(pkl_dir):
        return None, None, None

    hashes = {}
    for i in range(NUM_AGENTS):
        pkl_path = os.path.join(pkl_dir, f"{i}.pkl")
        if os.path.exists(pkl_path):
            h = md5_file(pkl_path)
            hashes[i] = h

    if not hashes:
        return None, None, None

    hash_to_agents = defaultdict(list)
    for agent_id, h in hashes.items():
        hash_to_agents[h].append(agent_id)

    num_unique = len(hash_to_agents)
    max_group = max(len(v) for v in hash_to_agents.values())
    return num_unique, max_group, dict(hash_to_agents)


def load_distance_vector(base_dir, iteration):
    """Load VAE latent distance vector for a given iteration."""
    dist_path = os.path.join(base_dir, "unimal_distance", str(iteration), "distance_vector.pt")
    if os.path.exists(dist_path):
        return torch.load(dist_path, map_location="cpu", weights_only=True)
    return None


def get_available_distance_iters(base_dir):
    """Get sorted list of iterations that have distance vectors."""
    dist_dir = os.path.join(base_dir, "unimal_distance")
    if not os.path.isdir(dist_dir):
        return []
    iters = []
    for name in os.listdir(dist_dir):
        try:
            iters.append(int(name))
        except ValueError:
            pass
    return sorted(iters)


def load_results_json(base_dir):
    """Load training reward results."""
    path = os.path.join(base_dir, "Unimal-v0_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Main analysis ──────────────────────────────────────────────────────────
print("=" * 70)
print("DIVERSITY COLLAPSE ANALYSIS")
print("=" * 70)

all_results = {}
summary_rows = []

for task in TASKS:
    for cluster_idx in range(NUM_CLUSTERS):
        base = get_base_dir(task, cluster_idx)
        if base is None:
            continue

        key = f"{task}_c{cluster_idx}"
        result = {"task": task, "cluster": cluster_idx}

        # 1. Count unique via PKL hash
        n_unique_pkl, max_group_pkl, groups_pkl = count_unique_pkl(base)
        result["unique_pkl"] = n_unique_pkl
        result["max_group_pkl"] = max_group_pkl

        # 2. Count unique via XML hash
        n_unique_xml, max_group_xml, groups_xml = count_unique_xml(base)
        result["unique_xml"] = n_unique_xml
        result["max_group_xml"] = max_group_xml

        # 3. Final mean distance in VAE latent space
        dist_vec = load_distance_vector(base, FINAL_ITER)
        if dist_vec is not None:
            result["final_mean_distance"] = dist_vec.mean().item()
            result["final_median_distance"] = dist_vec.median().item()
            result["final_min_distance"] = dist_vec.min().item()
        else:
            result["final_mean_distance"] = None

        # 4. Load final rewards per agent
        results_json = load_results_json(base)
        if results_json and groups_pkl:
            rewards = {}
            for agent_id_str, agent_data in results_json.items():
                try:
                    aid = int(agent_id_str)
                    r = agent_data["reward"]["reward"]
                    rewards[aid] = r[-1] if r else 0
                except (ValueError, KeyError, IndexError):
                    pass
            result["final_rewards"] = rewards

        all_results[key] = result

        n_unique = n_unique_pkl if n_unique_pkl is not None else n_unique_xml
        max_grp = max_group_pkl if max_group_pkl is not None else max_group_xml
        if n_unique is None:
            print(f"  {task:10s} c{cluster_idx:2d}: NO DATA")
            continue
        summary_rows.append({
            "task": task,
            "cluster": cluster_idx,
            "unique_morphologies": n_unique,
            "max_duplicate_group": max_grp,
            "mean_distance": result.get("final_mean_distance"),
        })

        status = f"  {task:10s} c{cluster_idx:2d}: {n_unique:2d}/20 unique"
        if max_grp:
            status += f" (largest group: {max_grp})"
        if result.get("final_mean_distance") is not None:
            status += f"  dist={result['final_mean_distance']:.4f}"
        print(status)

# ── Aggregate statistics ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for task in TASKS:
    task_rows = [r for r in summary_rows if r["task"] == task]
    if not task_rows:
        continue
    uniques = [r["unique_morphologies"] for r in task_rows if r["unique_morphologies"] is not None]
    max_grps = [r["max_duplicate_group"] for r in task_rows if r["max_duplicate_group"] is not None]
    dists = [r["mean_distance"] for r in task_rows if r["mean_distance"] is not None]

    print(f"\n{task.upper()} ({len(task_rows)} clusters):")
    if uniques:
        print(f"  Unique morphologies: mean={np.mean(uniques):.1f}, median={np.median(uniques):.1f}, "
              f"min={np.min(uniques)}, max={np.max(uniques)}")
    if max_grps:
        print(f"  Largest duplicate group: mean={np.mean(max_grps):.1f}, max={np.max(max_grps)}")
    if dists:
        print(f"  VAE mean distance: mean={np.mean(dists):.4f}, min={np.min(dists):.4f}, max={np.max(dists):.4f}")

# Overall
all_uniques = [r["unique_morphologies"] for r in summary_rows if r["unique_morphologies"] is not None]
print(f"\nOVERALL ({len(summary_rows)} cluster-task pairs):")
print(f"  Unique morphologies: mean={np.mean(all_uniques):.2f}, median={np.median(all_uniques):.1f}, "
      f"min={np.min(all_uniques)}, max={np.max(all_uniques)}")
print(f"  Pairs with ≤3 unique: {sum(1 for u in all_uniques if u <= 3)}/{len(all_uniques)}")
print(f"  Pairs with ≤5 unique: {sum(1 for u in all_uniques if u <= 5)}/{len(all_uniques)}")

# ── Save CSV ──────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "diversity_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task", "cluster", "unique_morphologies",
                                           "max_duplicate_group", "mean_distance"])
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\nSaved: {csv_path}")

# ── Save full JSON ────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "diversity_full_analysis.json")
# Convert groups to serializable format
serializable = {}
for key, result in all_results.items():
    r = {k: v for k, v in result.items() if k != "final_rewards"}
    serializable[key] = r
with open(json_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"Saved: {json_path}")


# ── PLOT 1: Heatmap of unique morphologies ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 10))
matrix = np.full((NUM_CLUSTERS, len(TASKS)), np.nan)
for r in summary_rows:
    col = TASKS.index(r["task"])
    matrix[r["cluster"], col] = r["unique_morphologies"]

im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=1, vmax=20)
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels(TASKS)
ax.set_yticks(range(NUM_CLUSTERS))
ax.set_yticklabels(range(NUM_CLUSTERS))
ax.set_xlabel("Task")
ax.set_ylabel("Cluster")
ax.set_title("Unique Morphologies in Final Elite Pool (out of 20)")
plt.colorbar(im, ax=ax, label="# Unique")

# Add text annotations
for i in range(NUM_CLUSTERS):
    for j in range(len(TASKS)):
        val = matrix[i, j]
        if not np.isnan(val):
            color = "white" if val <= 5 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center", color=color, fontsize=9)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "diversity_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {heatmap_path}")


# ── PLOT 2: Distribution of unique counts per task ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for i, task in enumerate(TASKS):
    task_uniques = [r["unique_morphologies"] for r in summary_rows
                    if r["task"] == task and r["unique_morphologies"] is not None]
    axes[i].hist(task_uniques, bins=range(1, 22), align="left", edgecolor="black", alpha=0.7,
                 color=["#2196F3", "#4CAF50", "#FF9800"][i])
    axes[i].set_title(f"{task}")
    axes[i].set_xlabel("Unique morphologies")
    axes[i].axvline(np.mean(task_uniques), color="red", linestyle="--", linewidth=2,
                    label=f"mean={np.mean(task_uniques):.1f}")
    axes[i].legend()
    axes[i].set_xlim(0, 21)
axes[0].set_ylabel("Count (clusters)")
fig.suptitle("Distribution of Unique Morphologies per Elite Pool", fontsize=14)
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "diversity_distribution.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {dist_path}")


# ── PLOT 3: Temporal evolution of mean distance ───────────────────────────
# Sample a few clusters per task to show trajectory
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ti, task in enumerate(TASKS):
    ax = axes[ti]
    plotted = 0
    for cluster_idx in range(NUM_CLUSTERS):
        base = get_base_dir(task, cluster_idx)
        if base is None:
            continue
        iters = get_available_distance_iters(base)
        if len(iters) < 5:
            continue

        mean_dists = []
        valid_iters = []
        for it in iters:
            dv = load_distance_vector(base, it)
            if dv is not None:
                mean_dists.append(dv.mean().item())
                valid_iters.append(it)

        if valid_iters:
            alpha = 0.3 if plotted > 3 else 0.7
            lw = 1.0 if plotted > 3 else 1.5
            ax.plot(valid_iters, mean_dists, alpha=alpha, linewidth=lw,
                    label=f"c{cluster_idx}" if plotted < 5 else None)
            plotted += 1

    ax.set_title(f"{task}")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean VAE latent distance")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle("Diversity Over Training (VAE Latent Mean Distance)", fontsize=14)
plt.tight_layout()
temporal_path = os.path.join(OUTPUT_DIR, "diversity_temporal.png")
plt.savefig(temporal_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {temporal_path}")


# ── PLOT 4: Duplicate group size distribution ─────────────────────────────
# For each cluster-task, show the group sizes (how many agents share the same morphology)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ti, task in enumerate(TASKS):
    all_group_sizes = []
    for cluster_idx in range(NUM_CLUSTERS):
        key = f"{task}_c{cluster_idx}"
        if key not in all_results:
            continue
        r = all_results[key]
        groups = r.get("groups_pkl") or r.get("groups_xml")
        # Reconstruct from unique count
        n_unique = r.get("unique_pkl") or r.get("unique_xml")
        max_grp = r.get("max_group_pkl") or r.get("max_group_xml")
        if n_unique is not None and max_grp is not None:
            all_group_sizes.append(max_grp)

    axes[ti].hist(all_group_sizes, bins=range(1, 22), align="left", edgecolor="black",
                  alpha=0.7, color=["#2196F3", "#4CAF50", "#FF9800"][ti])
    axes[ti].set_title(f"{task}")
    axes[ti].set_xlabel("Largest duplicate group size")
    if all_group_sizes:
        axes[ti].axvline(np.mean(all_group_sizes), color="red", linestyle="--",
                         label=f"mean={np.mean(all_group_sizes):.1f}")
        axes[ti].legend()
axes[0].set_ylabel("Count (clusters)")
fig.suptitle("Size of Largest Morphology Duplicate Group per Elite Pool", fontsize=14)
plt.tight_layout()
groups_path = os.path.join(OUTPUT_DIR, "diversity_max_group_size.png")
plt.savefig(groups_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {groups_path}")


# ── PLOT 5: Concrete example — one cluster-task pair ──────────────────────
# Pick a clear example of collapse to illustrate
worst_pair = min(summary_rows, key=lambda r: r["unique_morphologies"] if r["unique_morphologies"] else 20)
print(f"\nMost collapsed pair: {worst_pair['task']} c{worst_pair['cluster']} "
      f"({worst_pair['unique_morphologies']} unique)")

best_pair = max(summary_rows, key=lambda r: r["unique_morphologies"] if r["unique_morphologies"] else 0)
print(f"Most diverse pair:   {best_pair['task']} c{best_pair['cluster']} "
      f"({best_pair['unique_morphologies']} unique)")

# Show detailed grouping for worst pair
worst_key = f"{worst_pair['task']}_c{worst_pair['cluster']}"
worst_result = all_results[worst_key]

# Recompute groups with detail
base = get_base_dir(worst_pair["task"], worst_pair["cluster"])
_, _, groups = count_unique_pkl(base)
if groups:
    print(f"\nDetailed grouping for {worst_key}:")
    sorted_groups = sorted(groups.values(), key=len, reverse=True)
    for gi, group in enumerate(sorted_groups):
        print(f"  Group {gi+1} ({len(group)} agents): agents {group}")


# ── PLOT 6: Scatter unique vs mean_distance ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"ft": "#2196F3", "incline": "#4CAF50", "obstacle": "#FF9800"}
for r in summary_rows:
    if r["unique_morphologies"] is not None and r["mean_distance"] is not None:
        ax.scatter(r["unique_morphologies"], r["mean_distance"],
                   c=colors[r["task"]], alpha=0.7, s=60, edgecolors="black", linewidth=0.5)

# Legend
for task in TASKS:
    ax.scatter([], [], c=colors[task], label=task, s=60, edgecolors="black", linewidth=0.5)
ax.legend()
ax.set_xlabel("Unique morphologies (MD5 dedup)")
ax.set_ylabel("Mean VAE latent distance")
ax.set_title("Unique Count vs. VAE Distance Metric")
ax.grid(True, alpha=0.3)
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, "diversity_unique_vs_distance.png")
plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {scatter_path}")

print("\nDone!")
