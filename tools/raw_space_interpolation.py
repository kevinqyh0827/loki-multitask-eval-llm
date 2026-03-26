#!/usr/bin/env python3
"""Midpoint interpolation in raw design space ([11, 14] vectors).

Compares raw-space interpolation against VAE latent-space interpolation
for the top-2 unique elite morphologies in a cluster.

Interpolation strategy per feature type:
  - Continuous (cols 0-3): average (v1 + v2) / 2
  - Categorical (cols 4-7): round((v1 + v2) / 2) to nearest int
  - Joint bits (cols 8-9): AND / min (round down to fewer joints)
  - Torso mode (col 10): copy from higher-reward parent
  - Attach site (col 11): copy from higher-reward parent per limb
  - Structural (cols 12-13): use shorter morphology's structure (round down)

When parents have different limb counts, the result uses min(len1, len2).

Usage:
    MUJOCO_GL=egl python tools/raw_space_interpolation.py --new_task bump
    MUJOCO_GL=egl python tools/raw_space_interpolation.py --new_task bump --cluster 18 --similar_task obstacle
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
from tools.vec_to_morphology import (
    vec_to_xml_safe,
    _find_seq_len,
    _apply_mask_conventions,
    _repair_depth_sequence,
    _repair_joints,
)
from tools.xml_to_design_space import get_sequence_length, DESIGN_VECTOR_COLUMNS


def deduplicate_pool(pool):
    """Deduplicate elite pool by XML file content (MD5 hash)."""
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


def interpolate_raw(vec1, vec2, mask1, mask2, alpha=0.5,
                    reward1=0.0, reward2=0.0):
    """Interpolate two [11, 14] design vectors in raw space.

    Uses the shorter morphology's limb count (round down) and AND logic
    for joint bits to keep the result simpler and more likely valid.

    Args:
        vec1, vec2: numpy arrays of shape [11, 14]
        mask1, mask2: numpy arrays of shape [11, 14]
        alpha: interpolation weight (0.0 = vec1, 1.0 = vec2)
        reward1, reward2: rewards for tie-breaking (higher reward parent
                          is used for torso_mode and attach_site)

    Returns:
        (vec_interp, mask_interp) as numpy arrays of shape [11, 14]
    """
    vec1 = vec1.copy().astype(np.float64)
    vec2 = vec2.copy().astype(np.float64)

    seq_len1 = get_sequence_length(vec1)
    seq_len2 = get_sequence_length(vec2)

    # Round down: use the shorter morphology's limb count
    result_len = min(seq_len1, seq_len2)

    # Determine which parent has higher reward (for tie-breaking)
    better = 1 if reward1 >= reward2 else 2
    better_vec = vec1 if better == 1 else vec2

    # Initialize result with zeros
    result = np.zeros_like(vec1)
    result_mask = np.zeros_like(mask1)

    # Row 0: Torso — special handling
    # Continuous: density (col 3) — average
    result[0, 3] = (1 - alpha) * vec1[0, 3] + alpha * vec2[0, 3]
    # Torso mode (col 10) — copy from higher-reward parent
    result[0, 10] = better_vec[0, 10]
    # EOS: only if result_len == 1 (unlikely, min is 3 limbs)
    result[0, 12] = 1 if result_len == 1 else 0
    # Depth: always 0 for torso
    result[0, 13] = 0
    # Mask for torso
    result_mask[0, 3] = True   # density
    result_mask[0, 10] = True  # torso_mode
    result_mask[0, 12] = True  # EOS
    result_mask[0, 13] = True  # depth

    # Rows 1 to result_len-1: Limbs
    for i in range(1, result_len):
        # --- Continuous features (cols 0-3): weighted average ---
        for col in range(4):
            result[i, col] = (1 - alpha) * vec1[i, col] + alpha * vec2[i, col]

        # --- Categorical features (cols 4-7): round average ---
        for col in range(4, 8):
            avg = (1 - alpha) * vec1[i, col] + alpha * vec2[i, col]
            result[i, col] = round(avg)

        # --- Joint bits (cols 8-9): AND / min (round down) ---
        result[i, 8] = min(vec1[i, 8], vec2[i, 8])  # jx_bit
        result[i, 9] = min(vec1[i, 9], vec2[i, 9])  # jy_bit

        # --- Torso mode (col 10): always 0 for limbs ---
        result[i, 10] = 0

        # --- Attach site (col 11): copy from higher-reward parent ---
        result[i, 11] = better_vec[i, 11]

        # --- EOS (col 12): mark last limb ---
        result[i, 12] = 1 if i == result_len - 1 else 0

        # --- Depth (col 13): use shorter parent's depth structure ---
        # Pick depth from whichever parent has the shorter sequence
        shorter_vec = vec1 if seq_len1 <= seq_len2 else vec2
        result[i, 13] = shorter_vec[i, 13]

        # Build mask for this row
        jx_bit = int(round(result[i, 8]))
        jy_bit = int(round(result[i, 9]))
        result_mask[i, 0] = True   # orient_r
        result_mask[i, 1] = bool(jx_bit)  # jx_gear
        result_mask[i, 2] = bool(jy_bit)  # jy_gear
        result_mask[i, 3] = True   # density
        result_mask[i, 4] = True   # theta
        result_mask[i, 5] = True   # phi
        result_mask[i, 6] = bool(jx_bit)  # jx_range
        result_mask[i, 7] = bool(jy_bit)  # jy_range
        result_mask[i, 8] = True   # jx_bit
        result_mask[i, 9] = True   # jy_bit
        result_mask[i, 10] = False  # torso_mode (not for limbs)
        attach_on_torso = (result[i, 11] < 0)
        result_mask[i, 11] = not attach_on_torso
        result_mask[i, 12] = True  # EOS
        result_mask[i, 13] = True  # depth

    # Zero out padded rows
    result[result_len:, :] = 0
    result_mask[result_len:, :] = False

    # Validate and repair the depth sequence
    from tools.util import check_dfs_validity
    depths = result[:result_len, 13].astype(int).tolist()
    if not check_dfs_validity(depths):
        depths = _repair_depth_sequence(depths)
        for idx in range(result_len):
            result[idx, 13] = depths[idx]

    # Ensure every limb has at least one joint
    for i in range(1, result_len):
        if result[i, 8] == 0 and result[i, 9] == 0:
            # Round down AND gave no joints — give jx_bit from higher-reward parent
            result[i, 8] = 1
            result_mask[i, 1] = True
            result_mask[i, 6] = True
            # Use average gear/range from parents for the restored joint
            result[i, 1] = (1 - alpha) * vec1[i, 1] + alpha * vec2[i, 1]
            result[i, 6] = round((1 - alpha) * vec1[i, 6] + alpha * vec2[i, 6])

    # Apply mask conventions to clean up
    _apply_mask_conventions(result, result_len)

    return result, result_mask.astype(bool)


def main():
    parser = argparse.ArgumentParser(
        description="Midpoint Interpolation in Raw Design Space vs Latent Space"
    )
    parser.add_argument("--new_task", type=str, required=True,
                        help="New task name (must have LLM recommendation)")
    parser.add_argument("--cluster", type=int, default=None)
    parser.add_argument("--similar_task", type=str, default=None,
                        choices=list(TASK_DIRS.keys()))
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Interpolation weight (0.0 = elite1, 1.0 = elite2)")
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
    print("  Raw Design Space vs Latent Space Midpoint Interpolation")
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
    print(f"  Alpha: {args.alpha}")

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
        print("ERROR: Need at least 2 unique elites for interpolation.")
        sys.exit(1)

    top2 = unique_pool[:2]
    print(f"\n  Top-2 unique elites:")
    for i, a in enumerate(top2):
        seq_len = get_sequence_length(a["vec"].numpy())
        print(f"    #{i+1}: Agent {a['agent_id']}, reward={a['reward']:.1f}, "
              f"limbs={seq_len}")

    # --- Extract design vectors ---
    vec1 = top2[0]["vec"].numpy().astype(np.float64)
    vec2 = top2[1]["vec"].numpy().astype(np.float64)
    # Build masks from the vec (approximate: active if non-zero or structurally required)
    # We don't have ground-truth masks from the elite pool, so construct them
    seq1 = get_sequence_length(vec1)
    seq2 = get_sequence_length(vec2)
    mask1 = np.ones_like(vec1, dtype=bool)
    mask1[seq1:, :] = False
    mask2 = np.ones_like(vec2, dtype=bool)
    mask2[seq2:, :] = False

    # --- Raw-space interpolation ---
    print(f"\n--- Raw Design Space Interpolation ---")
    print(f"  Parent 1: {seq1} tokens, Parent 2: {seq2} tokens")
    result_len = min(seq1, seq2)
    print(f"  Result: {result_len} tokens (round down)")

    vec_raw, mask_raw = interpolate_raw(
        vec1, vec2, mask1, mask2,
        alpha=args.alpha,
        reward1=top2[0]["reward"],
        reward2=top2[1]["reward"],
    )

    # Convert raw-space result to XML
    vec_raw_tensor = torch.tensor(vec_raw, dtype=torch.float32)
    xml_raw = vec_to_xml_safe(vec_raw_tensor)
    raw_valid = xml_raw is not None
    print(f"  Raw-space midpoint: {'VALID' if raw_valid else 'INVALID'}")

    if raw_valid:
        raw_save = os.path.join(
            xml_dir, f"raw_midpoint_C{best_cluster}_{most_similar_task}_{args.new_task}.xml"
        )
        with open(raw_save, "w") as f:
            f.write(xml_raw)
        print(f"  Saved: {raw_save}")

    # --- Latent-space interpolation (for comparison) ---
    print(f"\n--- Latent Space Interpolation ---")
    z1 = top2[0]["z"]  # [1, 11, 32]
    z2 = top2[1]["z"]  # [1, 11, 32]
    z_mid = (1 - args.alpha) * z1 + args.alpha * z2
    l2_dist = torch.norm(z1 - z2).item()
    print(f"  L2 distance(z1, z2): {l2_dist:.4f}")

    decoded_vec = decode_from_latent(model, z_mid, device)
    xml_latent = vec_to_xml_safe(decoded_vec[0])
    latent_valid = xml_latent is not None
    print(f"  Latent-space midpoint: {'VALID' if latent_valid else 'INVALID'}")

    if latent_valid:
        latent_save = os.path.join(
            xml_dir, f"latent_midpoint_C{best_cluster}_{most_similar_task}_{args.new_task}.xml"
        )
        with open(latent_save, "w") as f:
            f.write(xml_latent)
        print(f"  Saved: {latent_save}")

    # --- Render all four morphologies ---
    print("\n--- Rendering ---")
    from PIL import Image

    images = []
    labels = []

    # Elite 1
    try:
        px1 = render_morphology_from_xml_path(
            top2[0]["xml_path"], args.width, args.height
        )
        r_str = f"R={top2[0]['reward']:.0f}" if top2[0]["reward"] > 0 else "R=?"
        px1 = add_text_to_image(px1, [
            f"Elite #1 ({top2[0]['agent_id']})",
            f"{r_str}, {seq1} limbs",
        ])
    except Exception as e:
        print(f"  Render failed for elite 1: {e}")
        px1 = create_invalid_placeholder(args.width, args.height)
    images.append(px1)
    labels.append(f"Elite #1 ({seq1} limbs)")

    # Elite 2
    try:
        px2 = render_morphology_from_xml_path(
            top2[1]["xml_path"], args.width, args.height
        )
        r_str = f"R={top2[1]['reward']:.0f}" if top2[1]["reward"] > 0 else "R=?"
        px2 = add_text_to_image(px2, [
            f"Elite #2 ({top2[1]['agent_id']})",
            f"{r_str}, {seq2} limbs",
        ])
    except Exception as e:
        print(f"  Render failed for elite 2: {e}")
        px2 = create_invalid_placeholder(args.width, args.height)
    images.append(px2)
    labels.append(f"Elite #2 ({seq2} limbs)")

    # Raw-space midpoint
    if raw_valid:
        try:
            px_raw = render_from_xml_string(xml_raw, args.width, args.height)
            px_raw = add_text_to_image(px_raw, [
                "Raw Midpoint",
                f"{result_len} limbs (min)",
            ])
        except Exception as e:
            print(f"  Render failed for raw midpoint: {e}")
            px_raw = create_invalid_placeholder(args.width, args.height)
            raw_valid = False
    else:
        px_raw = create_invalid_placeholder(args.width, args.height)
    images.append(px_raw)
    labels.append(f"Raw Midpoint ({result_len} limbs)")

    # Latent-space midpoint
    if latent_valid:
        try:
            px_lat = render_from_xml_string(xml_latent, args.width, args.height)
            latent_seq = get_sequence_length(decoded_vec[0].detach().cpu().numpy())
            px_lat = add_text_to_image(px_lat, [
                "Latent Midpoint",
                f"L2={l2_dist:.2f}, {latent_seq} limbs",
            ])
        except Exception as e:
            print(f"  Render failed for latent midpoint: {e}")
            px_lat = create_invalid_placeholder(args.width, args.height)
            latent_valid = False
    else:
        px_lat = create_invalid_placeholder(args.width, args.height)
    images.append(px_lat)
    labels.append("Latent Midpoint")

    # --- Composite visualization ---
    strip = create_strip_image(
        images,
        labels,
        strip_label=(
            f"C{best_cluster} x {most_similar_task} -> {args.new_task}: "
            f"Raw Design Space vs Latent Space Midpoint (alpha={args.alpha})"
        ),
        cell_width=args.width,
        cell_height=args.height,
    )

    out_path = os.path.join(
        args.output_dir,
        f"raw_vs_latent_C{best_cluster}_{most_similar_task}_{args.new_task}.png",
    )
    Image.fromarray(strip).save(out_path)
    print(f"\n  Visualization saved: {out_path}")

    # --- Save metadata ---
    meta = {
        "new_task": args.new_task,
        "best_cluster": best_cluster,
        "most_similar_task": most_similar_task,
        "alpha": args.alpha,
        "total_elites": len(pool),
        "unique_elites": len(unique_pool),
        "elite_1": {
            "agent_id": top2[0]["agent_id"],
            "reward": top2[0]["reward"],
            "sequence_length": seq1,
            "xml_path": top2[0]["xml_path"],
        },
        "elite_2": {
            "agent_id": top2[1]["agent_id"],
            "reward": top2[1]["reward"],
            "sequence_length": seq2,
            "xml_path": top2[1]["xml_path"],
        },
        "raw_midpoint": {
            "valid": raw_valid,
            "sequence_length": result_len,
            "strategy": "round_down_limbs_AND_joints",
        },
        "latent_midpoint": {
            "valid": latent_valid,
            "l2_distance": l2_dist,
        },
    }
    meta_path = os.path.join(
        args.output_dir,
        f"raw_vs_latent_C{best_cluster}_{most_similar_task}_{args.new_task}_metadata.json",
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata saved: {meta_path}")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
