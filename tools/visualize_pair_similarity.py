#!/usr/bin/env python3
"""Visualize a single task-pair similarity breakdown from the JSON results.

Usage:
    python tools/visualize_pair_similarity.py --task_a obstacle --task_b bump
    python tools/visualize_pair_similarity.py --task_a locomotion --task_b push_box_incline
    python tools/visualize_pair_similarity.py  # defaults to obstacle vs bump
"""

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_JSON = os.path.join(
    _PROJECT_ROOT, "results", "task_similarity",
    "similarity_matrix_20260319_221630.json",
)
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "task_similarity")

DIMENSION_LABELS = {
    "terrain": "Terrain",
    "reward_structure": "Reward Structure",
    "balance_requirements": "Balance Req.",
    "speed_vs_robustness": "Speed vs Robust.",
    "sensory_demands": "Sensory Demands",
    "contact_pattern": "Contact Pattern",
}

# Colors for each dimension bar
DIM_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]


def find_pair(data, task_a, task_b):
    """Find a specific pair from the parsed JSON (order-insensitive)."""
    for pair in data["parsed_pairs"]["task_pairs"]:
        a, b = pair["task_a"], pair["task_b"]
        if (a == task_a and b == task_b) or (a == task_b and b == task_a):
            return pair
    return None


def plot_pair_similarity(pair, output_path):
    """Create a visual breakdown of one task pair's similarity."""
    task_a = pair["task_a"]
    task_b = pair["task_b"]
    overall = pair["overall_similarity"]
    dims = pair["dimensions"]
    reasoning = pair["reasoning"]

    dim_keys = list(DIMENSION_LABELS.keys())
    dim_labels = [DIMENSION_LABELS[k] for k in dim_keys]
    dim_scores = [dims[k] for k in dim_keys]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1], width_ratios=[1.2, 1],
                          hspace=0.35, wspace=0.3)

    # ── Panel 1: Horizontal bar chart of 6 dimensions ──
    ax_bar = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(dim_keys))
    bars = ax_bar.barh(y_pos, dim_scores, height=0.6, color=DIM_COLORS,
                       edgecolor="white", linewidth=1.2)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(dim_labels, fontsize=11)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Similarity Score", fontsize=11)
    ax_bar.set_title(f"Dimension Scores", fontsize=13, fontweight="bold")
    ax_bar.invert_yaxis()

    # Annotate values on bars
    for bar, score in zip(bars, dim_scores):
        ax_bar.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f}", va="center", fontsize=11, fontweight="bold")

    # Add vertical line at overall score
    ax_bar.axvline(x=overall, color="#2c3e50", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"Overall = {overall:.2f}")
    ax_bar.legend(loc="lower right", fontsize=10)

    # ── Panel 2: Radar chart ──
    ax_radar = fig.add_subplot(gs[0, 1], projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(dim_keys), endpoint=False).tolist()
    scores_closed = dim_scores + [dim_scores[0]]
    angles_closed = angles + [angles[0]]

    ax_radar.fill(angles_closed, scores_closed, alpha=0.25, color="#3498db")
    ax_radar.plot(angles_closed, scores_closed, "o-", color="#2980b9",
                  linewidth=2, markersize=6)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels([l.replace(" ", "\n") for l in dim_labels],
                             fontsize=9)
    ax_radar.set_ylim(0, 1.0)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8,
                             color="grey")
    ax_radar.set_title("Radar Profile", fontsize=13, fontweight="bold",
                       pad=20)

    # ── Panel 3: Reasoning text box (bottom, spanning full width) ──
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    wrapped = textwrap.fill(reasoning, width=110)
    ax_text.text(0.02, 0.95, "LLM Reasoning:", fontsize=12, fontweight="bold",
                 transform=ax_text.transAxes, va="top",
                 family="sans-serif")
    ax_text.text(0.02, 0.75, wrapped, fontsize=10.5,
                 transform=ax_text.transAxes, va="top",
                 family="sans-serif", linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1",
                           edgecolor="#bdc3c7", alpha=0.9))

    # ── Suptitle ──
    fig.suptitle(
        f"{task_a.replace('_', ' ').title()}  vs  "
        f"{task_b.replace('_', ' ').title()}"
        f"    (Overall Similarity: {overall:.2f})",
        fontsize=15, fontweight="bold", y=0.98,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a single task-pair similarity breakdown")
    parser.add_argument("--json", type=str, default=_DEFAULT_JSON,
                        help="Path to similarity_matrix JSON")
    parser.add_argument("--task_a", type=str, default="obstacle")
    parser.add_argument("--task_b", type=str, default="bump")
    parser.add_argument("--output_dir", type=str, default=_OUTPUT_DIR)
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    pair = find_pair(data, args.task_a, args.task_b)
    if pair is None:
        print(f"ERROR: Pair ({args.task_a}, {args.task_b}) not found in JSON.")
        available = [(p["task_a"], p["task_b"])
                     for p in data["parsed_pairs"]["task_pairs"]]
        print(f"Available pairs: {available}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"pair_{args.task_a}_vs_{args.task_b}.png",
    )

    print(f"\n{'=' * 70}")
    print(f"  {args.task_a} vs {args.task_b}")
    print(f"  Overall Similarity: {pair['overall_similarity']:.2f}")
    print(f"{'=' * 70}")
    print(f"\n  Dimension Scores:")
    for dim, label in DIMENSION_LABELS.items():
        score = pair["dimensions"][dim]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {label:<18} {bar} {score:.2f}")
    print(f"\n  Reasoning:")
    for line in textwrap.wrap(pair["reasoning"], width=80):
        print(f"    {line}")
    print()

    plot_pair_similarity(pair, out_path)
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
