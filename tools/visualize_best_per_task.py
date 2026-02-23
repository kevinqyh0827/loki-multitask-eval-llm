#!/usr/bin/env python3
"""
For each task, visualize the top-3 performing clusters and their morphologies.

Usage:
    python tools/visualize_best_per_task.py

Output:
    results/task_best_clusters/{task}_top3_clusters.png
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description="Visualize best clusters per task")
    parser.add_argument("--table_path", type=str,
                        default="./results/cluster_task_table.json")
    parser.add_argument("--renders_dir", type=str,
                        default="./results/cluster_renders")
    parser.add_argument("--output_dir", type=str,
                        default="./results/task_best_clusters")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Max morphology renders per cluster")
    args = parser.parse_args()

    if not os.path.exists(args.table_path):
        print(f"Error: {args.table_path} not found. Run build_performance_table.py first.")
        return

    with open(args.table_path) as f:
        table = json.load(f)

    # Discover tasks from the table
    sample_cluster = next(iter(table.values()))
    tasks = [t for t in sample_cluster.keys()]
    print(f"Tasks found: {tasks}")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    CELL_W, CELL_H = 160, 130
    TITLE_H = 35
    ROW_LABEL_W = 180

    for task in tasks:
        # Rank clusters by reward on this task
        cluster_scores = {}
        for cluster_str, task_data in table.items():
            cluster = int(cluster_str)
            reward = task_data.get(task, {}).get("mean_reward")
            if reward is not None:
                cluster_scores[cluster] = reward

        if not cluster_scores:
            print(f"  {task}: no data, skipping")
            continue

        top3 = sorted(cluster_scores, key=cluster_scores.get, reverse=True)[:3]

        # Build composite image: title + 3 rows (top clusters) x N_SAMPLES cols
        img_w = ROW_LABEL_W + CELL_W * args.num_samples
        img_h = TITLE_H + CELL_H * 3
        img = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Title
        draw.rectangle([(0, 0), (img_w, TITLE_H)], fill=(40, 60, 120))
        draw.text((10, 8), f"Task: {task} | Top-3 Clusters: {top3}",
                  fill=(255, 255, 255), font=font)

        for row_i, cluster_label in enumerate(top3):
            y = TITLE_H + row_i * CELL_H
            score = cluster_scores[cluster_label]

            # Row label
            draw.text((5, y + 5),
                      f"Cluster {cluster_label}\n{score:.1f}",
                      fill=(0, 0, 0), font=font_small)

            # Separator
            draw.line([(0, y), (img_w, y)], fill=(200, 200, 200))

            # Morphology renders
            cluster_render_dir = os.path.join(
                args.renders_dir, f"cluster_{cluster_label}")
            if not os.path.exists(cluster_render_dir):
                continue

            png_files = sorted(
                f for f in os.listdir(cluster_render_dir) if f.endswith(".png"))
            for col_i, png_file in enumerate(png_files[:args.num_samples]):
                x = ROW_LABEL_W + col_i * CELL_W
                try:
                    m = Image.open(os.path.join(cluster_render_dir, png_file))
                    m = m.resize((CELL_W - 4, CELL_H - 4))
                    img.paste(m, (x + 2, y + 2))
                except Exception:
                    pass

        out_path = os.path.join(args.output_dir, f"{task}_top3_clusters.png")
        img.save(out_path)
        print(f"  Saved {out_path}")

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
