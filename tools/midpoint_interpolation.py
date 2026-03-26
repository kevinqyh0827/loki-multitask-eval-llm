#!/usr/bin/env python3
"""Midpoint Latent Interpolation between Top-2 Unique Elites.

For a new task, finds the top 2 unique (by XML content) highest-rewarded
morphologies from the elite pool, encodes them into VAE latent space,
computes the midpoint, decodes it back, and renders a comparison.

Usage:
    MUJOCO_GL=egl python tools/midpoint_interpolation.py --new_task bump
    MUJOCO_GL=egl python tools/midpoint_interpolation.py --new_task push_box_incline
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metamorph"))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from tools.interpolation_demo import (
    load_vae_model,
    load_task_elite_pool,
    load_recommendation,
    decode_from_latent,
    render_from_xml_string,
    render_morphology_from_xml_path,
    add_text_to_image,
    create_invalid_placeholder,
    create_strip_image,
    VAE_CHECKPOINT,
    OUTPUT_DIR,
    TASK_DIRS,
)
from tools.vec_to_morphology import vec_to_xml_safe


def deduplicate_pool(pool):
    """Deduplicate elite pool by XML file content (MD5 hash).

    Returns list of unique elites, keeping the one with highest reward
    for each duplicate group.
    """
    hash_groups = {}
    for agent in pool:
        with open(agent["xml_path"], "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h not in hash_groups:
            hash_groups[h] = agent
        elif agent["reward"] > hash_groups[h]["reward"]:
            hash_groups[h] = agent
    unique = list(hash_groups.values())
    unique.sort(key=lambda a: a["reward"], reverse=True)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Midpoint Latent Interpolation between Top-2 Unique Elites"
    )
    parser.add_argument("--new_task", type=str, required=True,
                        help="New task name (must have LLM recommendation)")
    parser.add_argument("--cluster", type=int, default=None)
    parser.add_argument("--similar_task", type=str, default=None,
                        choices=list(TASK_DIRS.keys()))
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Alternative data root (default: metamorph/output/loki)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    xml_dir = os.path.join(args.output_dir, "xmls")
    os.makedirs(xml_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("  Midpoint Latent Interpolation — Top-2 Unique Elites")
    print("=" * 70)

    # --- Determine cluster & similar task ---
    if args.cluster is not None and args.similar_task is not None:
        best_cluster = args.cluster
        most_similar_task = args.similar_task
    else:
        best_cluster_rec, similar_task_rec, _ = load_recommendation(
            args.new_task, _PROJECT_ROOT
        )
        best_cluster = args.cluster if args.cluster is not None else best_cluster_rec
        most_similar_task = (
            args.similar_task if args.similar_task is not None else similar_task_rec
        )
    print(f"  New task: {args.new_task}")
    print(f"  Source: Cluster {best_cluster} x {most_similar_task}")

    # --- Load VAE ---
    print("\nLoading VAE model...")
    model = load_vae_model(VAE_CHECKPOINT, device)
    print("  VAE loaded")

    # --- Load & encode elite pool ---
    print(f"\nLoading elite pool: C{best_cluster} x {most_similar_task}...")
    pool = load_task_elite_pool(best_cluster, most_similar_task, model, device,
                                data_root=args.data_root)
    print(f"  Total elites: {len(pool)}")

    # --- Deduplicate ---
    unique_pool = deduplicate_pool(pool)
    print(f"  Unique elites: {len(unique_pool)}")

    if len(unique_pool) < 2:
        print("ERROR: Need at least 2 unique elites for midpoint interpolation.")
        sys.exit(1)

    top2 = unique_pool[:2]
    print(f"\n  Top-2 unique elites:")
    for i, a in enumerate(top2):
        print(f"    #{i+1}: Agent {a['agent_id']}, reward={a['reward']:.1f}")

    # --- Compute midpoint in latent space ---
    z1 = top2[0]["z"]  # [1, 11, 32]
    z2 = top2[1]["z"]  # [1, 11, 32]
    z_mid = (z1 + z2) / 2.0  # [1, 11, 32]

    print(f"\n  z1 shape: {z1.shape}")
    print(f"  z2 shape: {z2.shape}")
    print(f"  z_mid shape: {z_mid.shape}")

    # Compute L2 distance between the two latent codes
    l2_dist = torch.norm(z1 - z2).item()
    print(f"  L2 distance(z1, z2): {l2_dist:.4f}")

    # --- Decode midpoint ---
    print("\nDecoding midpoint latent...")
    decoded_vec = decode_from_latent(model, z_mid, device)
    xml_mid = vec_to_xml_safe(decoded_vec[0])
    mid_valid = xml_mid is not None

    if mid_valid:
        mid_save = os.path.join(
            xml_dir, f"midpoint_C{best_cluster}_{most_similar_task}_{args.new_task}.xml"
        )
        with open(mid_save, "w") as f:
            f.write(xml_mid)
        print(f"  Midpoint XML saved: {mid_save}")
    else:
        print("  WARNING: Midpoint decoded to an invalid morphology")

    # --- Render all three ---
    print("\nRendering morphologies...")
    from PIL import Image

    images = []
    labels = []

    # Elite 1
    try:
        px1 = render_morphology_from_xml_path(
            top2[0]["xml_path"], args.width, args.height
        )
        r_str = f"R={top2[0]['reward']:.0f}" if top2[0]["reward"] > 0 else "R=?"
        px1 = add_text_to_image(px1, [f"Elite #1 (Agent {top2[0]['agent_id']})", r_str])
    except Exception as e:
        print(f"  Render failed for elite 1: {e}")
        px1 = create_invalid_placeholder(args.width, args.height)
    images.append(px1)
    labels.append(f"Elite #1 (Agent {top2[0]['agent_id']})")

    # Elite 2
    try:
        px2 = render_morphology_from_xml_path(
            top2[1]["xml_path"], args.width, args.height
        )
        r_str = f"R={top2[1]['reward']:.0f}" if top2[1]["reward"] > 0 else "R=?"
        px2 = add_text_to_image(px2, [f"Elite #2 (Agent {top2[1]['agent_id']})", r_str])
    except Exception as e:
        print(f"  Render failed for elite 2: {e}")
        px2 = create_invalid_placeholder(args.width, args.height)
    images.append(px2)
    labels.append(f"Elite #2 (Agent {top2[1]['agent_id']})")

    # Midpoint
    if mid_valid:
        try:
            px_mid = render_from_xml_string(xml_mid, args.width, args.height)
            px_mid = add_text_to_image(
                px_mid, ["Midpoint (z1+z2)/2", f"L2 dist={l2_dist:.2f}"]
            )
        except Exception as e:
            print(f"  Render failed for midpoint: {e}")
            px_mid = create_invalid_placeholder(args.width, args.height)
            mid_valid = False
    else:
        px_mid = create_invalid_placeholder(args.width, args.height)
    images.append(px_mid)
    labels.append("Midpoint (z1+z2)/2")

    # --- Composite visualization ---
    strip = create_strip_image(
        images,
        labels,
        strip_label=(
            f"C{best_cluster} x {most_similar_task} -> {args.new_task}: "
            f"Top-2 Unique Elites + Latent Midpoint"
        ),
        cell_width=args.width,
        cell_height=args.height,
    )

    out_path = os.path.join(
        args.output_dir,
        f"midpoint_C{best_cluster}_{most_similar_task}_{args.new_task}.png",
    )
    Image.fromarray(strip).save(out_path)
    print(f"\n  Visualization saved: {out_path}")

    # --- Save metadata ---
    meta = {
        "new_task": args.new_task,
        "best_cluster": best_cluster,
        "most_similar_task": most_similar_task,
        "total_elites": len(pool),
        "unique_elites": len(unique_pool),
        "elite_1": {
            "agent_id": top2[0]["agent_id"],
            "reward": top2[0]["reward"],
            "xml_path": top2[0]["xml_path"],
        },
        "elite_2": {
            "agent_id": top2[1]["agent_id"],
            "reward": top2[1]["reward"],
            "xml_path": top2[1]["xml_path"],
        },
        "latent_l2_distance": l2_dist,
        "midpoint_valid": mid_valid,
    }
    meta_path = os.path.join(
        args.output_dir,
        f"midpoint_C{best_cluster}_{most_similar_task}_{args.new_task}_metadata.json",
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata saved: {meta_path}")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
