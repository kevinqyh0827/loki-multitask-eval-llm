#!/usr/bin/env python3
"""Plot transfer efficiency curves: reward vs fine-tuning budget.

X-axis: fine-tuning steps (0 = zero-shot, then 2e6, 5e6, 1e7, 2e7, 5e7)
Y-axis: reward
Each cluster is a thin line; bold line = mean across clusters.
Horizontal dashed line = LOKI baseline (trained from scratch on target task).

Usage:
    # Old 20-cluster obstacle -> many_obstacle
    python scripts/transfer/plot_transfer_curve.py \
        --task_a obstacle --task_b many_obstacle \
        --transfer_dir metamorph/output/transfer \
        --loki_dir metamorph/output/loki \
        --num_clusters 20 --seed 3429

    # New 500K ft <-> incline
    python scripts/transfer/plot_transfer_curve.py \
        --task_a ft --task_b incline \
        --transfer_dir metamorph/output/transfer_500k \
        --loki_dir metamorph/output/loki_500k \
        --num_clusters 40 --seed 3429

    # Only one direction
    python scripts/transfer/plot_transfer_curve.py \
        --task_a obstacle --task_b many_obstacle --direction a2b \
        --transfer_dir metamorph/output/transfer \
        --loki_dir metamorph/output/loki \
        --num_clusters 20 --seed 3429
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


BUDGETS = ["2e6", "5e6", "1e7", "2e7", "5e7"]
BUDGET_STEPS = [0, 2e6, 5e6, 1e7, 2e7, 5e7]

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
    """Best agent tail-50 reward from LOKI training on target task."""
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
    """Zero-shot mean reward."""
    for fmt in [
        os.path.join(transfer_dir, "zero_shot", f"{src}_to_{tgt}",
                     f"c{cluster}", f"seed{seed}", "eval_results.json"),
        os.path.join(transfer_dir, "zero_shot", f"{src}_to_{tgt}",
                     f"c{cluster}", "eval_results.json"),
    ]:
        data = load_json(fmt)
        if data is not None:
            return data.get("mean_reward"), data.get("std_reward")
    return None, None


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


def collect_direction(transfer_dir, loki_dir, src, tgt, num_clusters, seed):
    """Collect all data for one transfer direction.

    Returns:
        cluster_curves: dict {cluster_id: [zs, ft_2e6, ft_5e6, ...]}
        baselines: dict {cluster_id: loki_baseline_reward}
    """
    cluster_curves = {}
    baselines = {}

    for c in range(num_clusters):
        sc = str(c)
        zs, _ = get_zero_shot_reward(transfer_dir, src, tgt, sc, seed)
        ft_vals = []
        for b in BUDGETS:
            ft_vals.append(get_finetune_reward(transfer_dir, src, tgt, sc, b, seed))

        # Only include clusters with at least zero-shot or one fine-tune result
        if zs is not None or any(v is not None for v in ft_vals):
            cluster_curves[c] = [zs] + ft_vals

        baseline = get_loki_baseline_reward(loki_dir, tgt, sc, num_clusters, seed)
        if baseline is not None:
            baselines[c] = baseline

    return cluster_curves, baselines


def plot_direction(ax, transfer_dir, loki_dir, src, tgt, num_clusters, seed):
    """Plot one direction on a given axes."""
    curves, baselines = collect_direction(
        transfer_dir, loki_dir, src, tgt, num_clusters, seed
    )

    if not curves:
        ax.set_title(f"{src} → {tgt}\n(no data)")
        return

    # Plot individual cluster curves (thin, transparent)
    all_ys = []
    for c, vals in sorted(curves.items()):
        xs, ys = [], []
        for i, v in enumerate(vals):
            if v is not None:
                xs.append(BUDGET_STEPS[i])
                ys.append(v)
        if xs:
            ax.plot(xs, ys, "o-", alpha=0.25, linewidth=0.8, markersize=3,
                    color="tab:blue")
        # Pad with NaN for mean calculation
        all_ys.append([v if v is not None else np.nan for v in vals])

    # Mean curve (bold)
    if all_ys:
        arr = np.array(all_ys)
        with np.errstate(all="ignore"):
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
            counts = np.sum(~np.isnan(arr), axis=0)

        valid = ~np.isnan(means)
        xs_valid = np.array(BUDGET_STEPS)[valid]
        means_valid = means[valid]
        stds_valid = stds[valid]

        ax.plot(xs_valid, means_valid, "o-", color="tab:blue", linewidth=2.5,
                markersize=6, label=f"Transfer mean (n={len(curves)})", zorder=5)
        ax.fill_between(xs_valid, means_valid - stds_valid, means_valid + stds_valid,
                        alpha=0.15, color="tab:blue")

    # LOKI baseline (horizontal dashed line)
    if baselines:
        baseline_mean = np.mean(list(baselines.values()))
        ax.axhline(baseline_mean, color="tab:red", linestyle="--", linewidth=2,
                   label=f"LOKI baseline on {tgt} ({baseline_mean:.0f})")

    ax.set_xlabel("Fine-tuning steps")
    ax.set_ylabel("Reward")
    ax.set_title(f"{src} → {tgt}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Log-like x-axis ticks
    ax.set_xticks(BUDGET_STEPS)
    ax.set_xticklabels(["0\n(zero-shot)", "2e6", "5e6", "1e7", "2e7", "5e7"],
                       fontsize=7)


def main():
    parser = argparse.ArgumentParser(
        description="Plot transfer efficiency curves"
    )
    parser.add_argument("--task_a", required=True)
    parser.add_argument("--task_b", required=True)
    parser.add_argument("--transfer_dir", default="metamorph/output/transfer_500k")
    parser.add_argument("--loki_dir", default="metamorph/output/loki_500k")
    parser.add_argument("--num_clusters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3429)
    parser.add_argument("--direction", choices=["both", "a2b", "b2a"], default="both",
                        help="Which direction(s) to plot")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: results/transfer_{a}_{b}.png)")
    args = parser.parse_args()

    plot_a2b = args.direction in ("both", "a2b")
    plot_b2a = args.direction in ("both", "b2a")

    if args.direction == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_a2b, ax_b2a = axes
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if plot_a2b:
        target_ax = axes[0] if args.direction == "both" else ax
        plot_direction(target_ax, args.transfer_dir, args.loki_dir,
                       args.task_a, args.task_b, args.num_clusters, args.seed)

    if plot_b2a:
        target_ax = axes[1] if args.direction == "both" else ax
        plot_direction(target_ax, args.transfer_dir, args.loki_dir,
                       args.task_b, args.task_a, args.num_clusters, args.seed)

    fig.suptitle(
        f"Transfer Efficiency: {args.task_a} ↔ {args.task_b}\n"
        f"({args.num_clusters} clusters, seed {args.seed})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = args.out or f"results/transfer_{args.task_a}_{args.task_b}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Also save as PDF for paper
    pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
