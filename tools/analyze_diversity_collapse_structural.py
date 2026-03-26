#!/usr/bin/env python3
"""Analyze diversity collapse using STRUCTURAL morphology similarity.

Unlike byte-level MD5, this uses geom_orientation + Hungarian matching
(the same method used in the LOKI codebase) to determine if two morphologies
are structurally the same. Two morphologies that differ only in minor
continuous parameters (joint angles, limb sizes) but share the same topology
are counted as identical.

This should reproduce the ~3.6 unique morphologies finding.
"""

import os
import sys
import json
import hashlib
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["LOKI_SEQUENTIAL_SIMILARITY"] = "1"  # avoid fork bombs
os.environ.setdefault("MUJOCO_GL", "egl")

# ── Inline structural similarity functions (from derl/utils/similarity.py) ──
# Avoids fragile derl import chain
from lxml import etree
import mujoco
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def geom_orientations_from_xml(path):
    """Get geom orientations for all limbs in a morphology XML.
    Uses the new mujoco Python bindings (mujoco >= 2.3).
    """
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    orientations = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.startswith("limb/"):
            # geom_xmat is [ngeom, 9] — rotation matrix flattened
            geom_frame = data.geom_xmat[i].copy().ravel()
            orientations.append(geom_frame)

    return orientations


def is_same_morphology(m1, m2):
    """Check if two morphologies are structurally the same.
    Uses Hungarian matching on geom orientation matrices.
    eps=0.2 for 9-dim geom orientations.
    """
    if len(m1) != len(m2):
        return False
    if len(m1) == 0:
        return True
    cost = cdist(m1, m2)
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment_cost = cost[row_ind, col_ind].sum()
    eps = 0.2 if len(m1[0]) == 9 else 1e-3
    return assignment_cost < eps


def count_structural_unique_via_graph(xml_paths):
    """Count structurally unique morphologies using all-pairs comparison.
    Returns (num_unique, groups_list) where groups_list is sorted by size desc.
    """
    # Get orientations for each agent
    orientations = {}
    for path in xml_paths:
        agent_id = os.path.splitext(os.path.basename(path))[0]
        try:
            ori = geom_orientations_from_xml(path)
            orientations[agent_id] = ori
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")

    if not orientations:
        return None, None

    # Build similarity graph
    G = nx.Graph()
    G.add_nodes_from(orientations.keys())

    agents = list(orientations.keys())
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if is_same_morphology(orientations[agents[i]], orientations[agents[j]]):
                G.add_edge(agents[i], agents[j])

    cc = list(nx.connected_components(G))
    num_unique = len(cc)

    # Sort groups by size descending
    groups = [sorted(list(comp)) for comp in sorted(cc, key=len, reverse=True)]
    return num_unique, groups


def count_structural_unique_hashed(xml_paths):
    """Fast hash-based structural uniqueness (less accurate, no MuJoCo needed)."""
    orientations = {}
    for path in xml_paths:
        agent_id = os.path.splitext(os.path.basename(path))[0]
        try:
            ori = geom_orientations_from_xml(path)
            orientations[agent_id] = ori
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")

    if not orientations:
        return None, None

    # Group by limb count, then hash
    limb_groups = defaultdict(list)
    for uid, ori in orientations.items():
        limb_groups[len(ori)].append(uid)

    G = nx.Graph()
    G.add_nodes_from(orientations.keys())

    for limb_count, uids in limb_groups.items():
        buckets = defaultdict(list)
        for uid in uids:
            sorted_oris = sorted(
                [tuple(round(v, 1) for v in o) for o in orientations[uid]]
            )
            canonical = hashlib.sha256(str(sorted_oris).encode()).hexdigest()
            buckets[canonical].append(uid)

        for bucket_uids in buckets.values():
            if len(bucket_uids) > 1:
                for i in range(1, len(bucket_uids)):
                    G.add_edge(bucket_uids[0], bucket_uids[i])

    cc = list(nx.connected_components(G))
    groups = [sorted(list(comp)) for comp in sorted(cc, key=len, reverse=True)]
    return len(cc), groups

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
    """Find the base directory for a cluster-task pair."""
    eval_path = os.path.join(EVAL_ROOT, task, "kmeans_cluster", str(NUM_CLUSTERS),
                             str(cluster_idx), SUFFIX)
    if os.path.isdir(eval_path):
        return eval_path
    train_path = os.path.join(TRAIN_ROOT, task, "kmeans_cluster", str(NUM_CLUSTERS),
                              str(cluster_idx), SUFFIX)
    if os.path.isdir(train_path):
        return train_path
    return None


def get_xml_paths(base_dir, iteration=FINAL_ITER):
    """Get paths to the 20 agent XML files for a given iteration."""
    xml_dir = os.path.join(base_dir, "xml_step", str(iteration))
    if not os.path.isdir(xml_dir):
        return []
    paths = []
    for i in range(NUM_AGENTS):
        p = os.path.join(xml_dir, f"{i}.xml")
        if os.path.exists(p):
            paths.append(p)
    return paths


    # (structural unique functions defined above as inlined functions)


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


# ── Main analysis ──────────────────────────────────────────────────────────
print("=" * 70)
print("STRUCTURAL DIVERSITY COLLAPSE ANALYSIS")
print("(using geom_orientation + Hungarian matching)")
print("=" * 70)

all_results = {}
summary_rows = []

for task in TASKS:
    print(f"\n--- {task.upper()} ---")
    for cluster_idx in range(NUM_CLUSTERS):
        base = get_base_dir(task, cluster_idx)
        if base is None:
            continue

        key = f"{task}_c{cluster_idx}"
        result = {"task": task, "cluster": cluster_idx}

        xml_paths = get_xml_paths(base)
        if not xml_paths:
            print(f"  c{cluster_idx:2d}: NO XML DATA")
            continue

        # Use hash-based structural comparison (fast)
        n_unique_hash, groups_hash = count_structural_unique_hashed(xml_paths)
        result["unique_hash"] = n_unique_hash

        # Also run all-pairs comparison for accuracy on the 20 agents (small enough)
        n_unique_allpairs, groups_allpairs = count_structural_unique_via_graph(xml_paths)
        result["unique_allpairs"] = n_unique_allpairs

        # Group sizes for allpairs (more accurate)
        if groups_allpairs:
            group_sizes = [len(g) for g in groups_allpairs]
            result["group_sizes"] = sorted(group_sizes, reverse=True)
            result["max_group"] = max(group_sizes)
        else:
            result["group_sizes"] = []
            result["max_group"] = None

        # VAE distance
        dist_vec = load_distance_vector(base, FINAL_ITER)
        if dist_vec is not None:
            result["mean_distance"] = dist_vec.mean().item()

        all_results[key] = result

        summary_rows.append({
            "task": task,
            "cluster": cluster_idx,
            "unique_hash": n_unique_hash,
            "unique_allpairs": n_unique_allpairs,
            "max_group": result.get("max_group"),
            "group_sizes": str(result.get("group_sizes", [])),
            "mean_distance": result.get("mean_distance"),
        })

        gs_str = str(result.get("group_sizes", [])[:5])
        print(f"  c{cluster_idx:2d}: hash={n_unique_hash:2d}/20  allpairs={n_unique_allpairs:2d}/20  "
              f"groups={gs_str}")


# ── Aggregate statistics ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY STATISTICS (all-pairs method)")
print("=" * 70)

for task in TASKS:
    task_rows = [r for r in summary_rows if r["task"] == task]
    if not task_rows:
        continue
    uniques = [r["unique_allpairs"] for r in task_rows if r["unique_allpairs"] is not None]
    max_grps = [r["max_group"] for r in task_rows if r["max_group"] is not None]

    print(f"\n{task.upper()} ({len(task_rows)} clusters):")
    if uniques:
        print(f"  Structural unique: mean={np.mean(uniques):.2f}, median={np.median(uniques):.1f}, "
              f"min={np.min(uniques)}, max={np.max(uniques)}")
    if max_grps:
        print(f"  Largest group: mean={np.mean(max_grps):.1f}, max={np.max(max_grps)}")

all_uniques = [r["unique_allpairs"] for r in summary_rows if r["unique_allpairs"] is not None]
all_max_grps = [r["max_group"] for r in summary_rows if r["max_group"] is not None]
print(f"\nOVERALL ({len(summary_rows)} cluster-task pairs):")
print(f"  Structural unique: mean={np.mean(all_uniques):.2f}, median={np.median(all_uniques):.1f}, "
      f"min={np.min(all_uniques)}, max={np.max(all_uniques)}")
print(f"  Pairs with ≤3 unique: {sum(1 for u in all_uniques if u <= 3)}/{len(all_uniques)}")
print(f"  Pairs with ≤5 unique: {sum(1 for u in all_uniques if u <= 5)}/{len(all_uniques)}")
print(f"  Pairs with ≤10 unique: {sum(1 for u in all_uniques if u <= 10)}/{len(all_uniques)}")

# ── Save CSV ──────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "structural_diversity_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task", "cluster", "unique_hash",
                                           "unique_allpairs", "max_group",
                                           "group_sizes", "mean_distance"])
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\nSaved: {csv_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "structural_diversity_full.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved: {json_path}")


# ── PLOT 1: Heatmap of structural unique morphologies ─────────────────────
fig, ax = plt.subplots(figsize=(8, 10))
matrix = np.full((NUM_CLUSTERS, len(TASKS)), np.nan)
for r in summary_rows:
    col = TASKS.index(r["task"])
    matrix[r["cluster"], col] = r["unique_allpairs"]

im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=1, vmax=20)
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels(TASKS)
ax.set_yticks(range(NUM_CLUSTERS))
ax.set_yticklabels(range(NUM_CLUSTERS))
ax.set_xlabel("Task")
ax.set_ylabel("Cluster")
ax.set_title("Structurally Unique Morphologies in Final Elite Pool\n(geom_orientation + Hungarian matching)")
plt.colorbar(im, ax=ax, label="# Unique")

for i in range(NUM_CLUSTERS):
    for j in range(len(TASKS)):
        val = matrix[i, j]
        if not np.isnan(val):
            color = "white" if val <= 5 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "structural_diversity_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {heatmap_path}")


# ── PLOT 2: Distribution per task ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
task_colors = {"ft": "#2196F3", "incline": "#4CAF50", "obstacle": "#FF9800"}
for i, task in enumerate(TASKS):
    task_uniques = [r["unique_allpairs"] for r in summary_rows
                    if r["task"] == task and r["unique_allpairs"] is not None]
    axes[i].hist(task_uniques, bins=range(1, 22), align="left", edgecolor="black", alpha=0.7,
                 color=task_colors[task])
    axes[i].set_title(f"{task}")
    axes[i].set_xlabel("Structurally unique morphologies")
    if task_uniques:
        axes[i].axvline(np.mean(task_uniques), color="red", linestyle="--", linewidth=2,
                        label=f"mean={np.mean(task_uniques):.1f}")
        axes[i].legend()
    axes[i].set_xlim(0, 21)
axes[0].set_ylabel("Count (clusters)")
fig.suptitle("Distribution of Structurally Unique Morphologies per Elite Pool", fontsize=14)
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "structural_diversity_distribution.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {dist_path}")


# ── PLOT 3: Group size breakdown for worst examples ──────────────────────
# Show top-10 most collapsed pairs as stacked bars
collapsed = sorted(summary_rows, key=lambda r: r["unique_allpairs"] if r["unique_allpairs"] else 20)[:15]

fig, ax = plt.subplots(figsize=(14, 6))
labels = [f"{r['task']}_c{r['cluster']}" for r in collapsed]
group_sizes_list = [all_results[f"{r['task']}_c{r['cluster']}"].get("group_sizes", []) for r in collapsed]

# Stacked bar: each group is a bar segment
max_groups = max(len(gs) for gs in group_sizes_list) if group_sizes_list else 1
cmap = plt.cm.Set3
for gi in range(max_groups):
    bottoms = []
    heights = []
    for gs in group_sizes_list:
        bottom = sum(gs[:gi]) if gi < len(gs) else sum(gs)
        height = gs[gi] if gi < len(gs) else 0
        bottoms.append(bottom)
        heights.append(height)
    ax.bar(labels, heights, bottom=bottoms, color=cmap(gi % 12), edgecolor="black", linewidth=0.5,
           label=f"Group {gi+1}" if gi < 6 else None)

ax.set_ylabel("Number of agents")
ax.set_xlabel("Cluster-Task pair")
ax.set_title("Elite Pool Composition (Most Collapsed Pairs)\nEach color = one structurally unique morphology")
ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
breakdown_path = os.path.join(OUTPUT_DIR, "structural_diversity_breakdown.png")
plt.savefig(breakdown_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {breakdown_path}")


# ── PLOT 4: Temporal diversity evolution (structural + VAE) ──────────────
# For a few cluster-task pairs, trace structural uniqueness over training
# This is expensive (MuJoCo sim per iteration), so sample sparsely
print("\nComputing temporal structural diversity for selected pairs...")
temporal_data = {}

# Pick 3 pairs: one per task, choose ones with interesting collapse
sample_pairs = []
for task in TASKS:
    task_rows_sorted = sorted(
        [r for r in summary_rows if r["task"] == task and r["unique_allpairs"] is not None],
        key=lambda r: r["unique_allpairs"]
    )
    if task_rows_sorted:
        sample_pairs.append((task, task_rows_sorted[0]["cluster"]))  # most collapsed per task

for task, cidx in sample_pairs:
    base = get_base_dir(task, cidx)
    if base is None:
        continue

    xml_base = os.path.join(base, "xml_step")
    if not os.path.isdir(xml_base):
        continue

    available_iters = sorted(int(d) for d in os.listdir(xml_base) if d.isdigit())
    # Sample ~15 iterations evenly
    if len(available_iters) > 15:
        step = len(available_iters) // 15
        sampled_iters = available_iters[::step]
        if available_iters[-1] not in sampled_iters:
            sampled_iters.append(available_iters[-1])
    else:
        sampled_iters = available_iters

    key = f"{task}_c{cidx}"
    temporal_data[key] = {"iters": [], "unique": [], "vae_dist": []}
    print(f"  {key}: checking {len(sampled_iters)} iterations...")

    for it in sampled_iters:
        xml_paths = get_xml_paths(base, iteration=it)
        if len(xml_paths) < NUM_AGENTS:
            continue
        try:
            n_uniq, _ = count_structural_unique_via_graph(xml_paths)
            dist_vec = load_distance_vector(base, it)
            mean_dist = dist_vec.mean().item() if dist_vec is not None else None

            temporal_data[key]["iters"].append(it)
            temporal_data[key]["unique"].append(n_uniq)
            temporal_data[key]["vae_dist"].append(mean_dist)
        except Exception as e:
            print(f"    iter {it}: {e}")

if temporal_data:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors_cycle = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    for i, (key, data) in enumerate(temporal_data.items()):
        color = colors_cycle[i % len(colors_cycle)]
        ax1.plot(data["iters"], data["unique"], marker="o", markersize=3,
                 color=color, label=key, linewidth=1.5)
        valid_dist = [(it, d) for it, d in zip(data["iters"], data["vae_dist"]) if d is not None]
        if valid_dist:
            ax2.plot([x[0] for x in valid_dist], [x[1] for x in valid_dist],
                     marker="o", markersize=3, color=color, label=key, linewidth=1.5)

    ax1.set_ylabel("Structurally unique morphologies")
    ax1.set_title("Diversity Collapse Over Training")
    ax1.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="max (20)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 22)

    ax2.set_ylabel("Mean VAE latent distance")
    ax2.set_xlabel("Training iteration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    temporal_path = os.path.join(OUTPUT_DIR, "structural_diversity_temporal.png")
    plt.savefig(temporal_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {temporal_path}")

# Save temporal data
temporal_json = os.path.join(OUTPUT_DIR, "structural_diversity_temporal.json")
with open(temporal_json, "w") as f:
    json.dump(temporal_data, f, indent=2)
print(f"Saved: {temporal_json}")

print("\nDone!")
