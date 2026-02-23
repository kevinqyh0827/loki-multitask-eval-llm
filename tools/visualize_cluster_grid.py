#!/usr/bin/env python3
"""
Create a composite grid image of cluster morphologies.

Rows = clusters, Columns = sampled morphologies.
Annotated with cluster labels and best-task info from the performance table.

Usage:
    python tools/visualize_cluster_grid.py [--num_clusters 20] [--num_samples 5]

Output:
    results/cluster_morphology_grid.png
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description="Create cluster morphology grid")
    parser.add_argument("--renders_dir", type=str,
                        default="./results/cluster_renders")
    parser.add_argument("--table_path", type=str,
                        default="./results/cluster_task_table.json",
                        help="Performance table JSON (optional, for annotations)")
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="./results/cluster_morphology_grid.png")
    parser.add_argument("--cell_w", type=int, default=200)
    parser.add_argument("--cell_h", type=int, default=160)
    args = parser.parse_args()

    CELL_W = args.cell_w
    CELL_H = args.cell_h
    LABEL_W = 220  # Width for the label column

    # Load performance table if available
    perf_table = {}
    if os.path.exists(args.table_path):
        with open(args.table_path) as f:
            raw = json.load(f)
        for cluster_str, task_data in raw.items():
            cluster = int(cluster_str)
            scores = {}
            for task, data in task_data.items():
                if data["mean_reward"] is not None:
                    scores[task] = data["mean_reward"]
            if scores:
                perf_table[cluster] = scores

    # Build grid
    grid_w = LABEL_W + CELL_W * args.num_samples
    grid_h = CELL_H * args.num_clusters
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    for cluster_label in range(args.num_clusters):
        y_offset = cluster_label * CELL_H

        # Draw label
        label_text = f"Cluster {cluster_label}"
        if cluster_label in perf_table:
            scores = perf_table[cluster_label]
            best_task = max(scores, key=scores.get)
            best_score = scores[best_task]
            label_text += f"\nBest: {best_task}"
            label_text += f" ({best_score:.1f})"
        draw.text((5, y_offset + 5), label_text, fill=(0, 0, 0), font=font)

        # Draw separator line
        draw.line([(0, y_offset), (grid_w, y_offset)], fill=(200, 200, 200))

        # Draw morphology renders
        cluster_render_dir = os.path.join(args.renders_dir, f"cluster_{cluster_label}")
        if not os.path.exists(cluster_render_dir):
            draw.text((LABEL_W + 10, y_offset + CELL_H // 2 - 6),
                      "(no renders)", fill=(180, 180, 180), font=font_small)
            continue

        png_files = sorted(f for f in os.listdir(cluster_render_dir) if f.endswith(".png"))
        for i, png_file in enumerate(png_files[:args.num_samples]):
            x_offset = LABEL_W + i * CELL_W
            img_path = os.path.join(cluster_render_dir, png_file)
            try:
                morph_img = Image.open(img_path).resize((CELL_W - 4, CELL_H - 4))
                grid_img.paste(morph_img, (x_offset + 2, y_offset + 2))
            except Exception as e:
                draw.text((x_offset + 5, y_offset + 5),
                          f"err", fill=(255, 0, 0), font=font_small)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    grid_img.save(args.output)
    print(f"Saved grid to {args.output}")
    print(f"Grid size: {grid_w} x {grid_h} pixels")


if __name__ == "__main__":
    main()
