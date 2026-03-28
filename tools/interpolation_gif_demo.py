#!/usr/bin/env python3
"""Interpolation GIF Demo — Smooth morphing between two elite morphologies.

Generates an animated GIF showing the continuous interpolation from
Elite #1 (alpha=0) to Elite #2 (alpha=1) in both VAE latent space
and raw design space, side by side.

Usage:
    # Using saved recommendation
    MUJOCO_GL=egl python tools/interpolation_gif_demo.py --cluster 18 --similar_task incline

    # Custom alpha steps
    MUJOCO_GL=egl python tools/interpolation_gif_demo.py --cluster 18 --similar_task obstacle --steps 21 --fps 5
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "metamorph"))

# ---------------------------------------------------------------------------
# Imports from existing tools
# ---------------------------------------------------------------------------
from tools.interpolation_demo import (
    load_vae_model,
    load_task_elite_pool,
    load_recommendation,
    decode_from_latent,
    render_from_xml_string,
    render_morphology_from_xml_path,
    add_text_to_image,
    create_invalid_placeholder,
    VAE_CHECKPOINT,
    TASK_DIRS,
)
from tools.midpoint_interpolation import deduplicate_pool
from tools.raw_space_interpolation import interpolate_raw
from tools.xml_to_design_space import get_sequence_length
from tools.vec_to_morphology import vec_to_xml_safe

_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "interpolation_gif")


def render_frame(xml_str, width, height, label_lines):
    """Render a morphology and add label text. Returns numpy array."""
    if xml_str is None:
        img = create_invalid_placeholder(width, height)
    else:
        try:
            img = render_from_xml_string(xml_str, width, height)
        except Exception as e:
            print(f"    Render failed: {e}")
            img = create_invalid_placeholder(width, height)
    return add_text_to_image(img, label_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Interpolation GIF Demo — Smooth morphing between elites"
    )
    parser.add_argument("--new_task", type=str, default="manipulation_ball",
                        help="Task name (for loading recommendation)")
    parser.add_argument("--cluster", type=int, required=True,
                        help="Cluster ID")
    parser.add_argument("--similar_task", type=str, required=True,
                        choices=list(TASK_DIRS.keys()),
                        help="Source task for elite pool")
    parser.add_argument("--steps", type=int, default=21,
                        help="Number of interpolation steps (default: 21)")
    parser.add_argument("--fps", type=int, default=4,
                        help="GIF frames per second (default: 4)")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["latent", "raw", "both"],
                        help="Interpolation mode: latent, raw, or both side-by-side")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    print("=" * 70)
    print("  Interpolation GIF Demo")
    print("=" * 70)
    print(f"  Cluster {args.cluster} x {args.similar_task}")
    print(f"  Steps: {args.steps}, FPS: {args.fps}, Mode: {args.mode}")

    # --- Load VAE ---
    print("\nLoading VAE model...")
    model = load_vae_model(VAE_CHECKPOINT, device)

    # --- Load & dedup elite pool ---
    print(f"Loading elite pool: C{args.cluster} x {args.similar_task}...")
    pool = load_task_elite_pool(
        args.cluster, args.similar_task, model, device,
        data_root=args.data_root,
    )
    unique_pool = deduplicate_pool(pool)
    print(f"  {len(pool)} total, {len(unique_pool)} unique")

    if len(unique_pool) < 2:
        print("ERROR: Need at least 2 unique elites.")
        sys.exit(1)

    top2 = unique_pool[:2]
    seq1 = get_sequence_length(top2[0]["vec"].numpy())
    seq2 = get_sequence_length(top2[1]["vec"].numpy())
    print(f"  Elite #1: Agent {top2[0]['agent_id']}, "
          f"R={top2[0]['reward']:.1f}, {seq1} limbs")
    print(f"  Elite #2: Agent {top2[1]['agent_id']}, "
          f"R={top2[1]['reward']:.1f}, {seq2} limbs")

    # --- Prepare interpolation data ---
    z1 = top2[0]["z"]  # [1, 11, 32]
    z2 = top2[1]["z"]

    vec1 = top2[0]["vec"].numpy().astype(np.float64)
    vec2 = top2[1]["vec"].numpy().astype(np.float64)
    mask1 = np.ones_like(vec1, dtype=bool)
    mask1[seq1:, :] = False
    mask2 = np.ones_like(vec2, dtype=bool)
    mask2[seq2:, :] = False
    result_len = min(seq1, seq2)

    l2_dist = torch.norm(z1 - z2).item()
    print(f"  L2 distance: {l2_dist:.4f}")

    # --- Generate alpha values (0 → 1 → 0 for loop, or just 0 → 1) ---
    alphas = np.linspace(0.0, 1.0, args.steps)

    # --- Generate frames ---
    print(f"\nGenerating {len(alphas)} frames...")
    from PIL import Image

    frames = []
    W, H = args.width, args.height

    for i, alpha in enumerate(alphas):
        print(f"  Frame {i+1}/{len(alphas)}: alpha={alpha:.2f}", end="")

        frame_images = []

        # --- Latent interpolation ---
        if args.mode in ("latent", "both"):
            z_interp = (1 - alpha) * z1 + alpha * z2
            decoded = decode_from_latent(model, z_interp, device)
            xml_lat = vec_to_xml_safe(decoded[0])
            lat_img = render_frame(xml_lat, W, H, [
                f"Latent (alpha={alpha:.2f})",
            ])
            frame_images.append(lat_img)

        # --- Raw interpolation ---
        if args.mode in ("raw", "both"):
            vec_r, _ = interpolate_raw(
                vec1, vec2, mask1, mask2,
                alpha=alpha,
                reward1=top2[0]["reward"],
                reward2=top2[1]["reward"],
            )
            xml_raw = vec_to_xml_safe(torch.tensor(vec_r, dtype=torch.float32))
            raw_img = render_frame(xml_raw, W, H, [
                f"Raw (alpha={alpha:.2f})",
            ])
            frame_images.append(raw_img)

        # Combine side by side if both
        if len(frame_images) == 2:
            frame = np.concatenate(frame_images, axis=1)
        else:
            frame = frame_images[0]

        frames.append(Image.fromarray(frame))
        print(" OK")

    # --- Also add reversed frames for smooth looping ---
    loop_frames = frames + frames[-2:0:-1]  # forward + reverse (no duplicate endpoints)

    # --- Save GIF ---
    gif_name = f"interpolation_C{args.cluster}_{args.similar_task}_{args.mode}_{timestamp}.gif"
    gif_path = os.path.join(args.output_dir, gif_name)
    duration_ms = int(1000 / args.fps)
    loop_frames[0].save(
        gif_path,
        save_all=True,
        append_images=loop_frames[1:],
        duration=duration_ms,
        loop=0,  # infinite loop
    )
    print(f"\n  GIF saved: {gif_path}")
    print(f"  Frames: {len(loop_frames)} ({len(alphas)} forward + "
          f"{len(loop_frames) - len(alphas)} reverse)")
    print(f"  Duration: {len(loop_frames) * duration_ms / 1000:.1f}s per loop")

    # --- Also save first, middle, last as individual PNGs ---
    key_frames = {
        "alpha_0.0": frames[0],
        "alpha_0.5": frames[len(frames)//2],
        "alpha_1.0": frames[-1],
    }
    for label, img in key_frames.items():
        png_path = os.path.join(
            args.output_dir,
            f"keyframe_C{args.cluster}_{args.similar_task}_{label}_{timestamp}.png"
        )
        img.save(png_path)

    # --- Save metadata ---
    meta = {
        "timestamp": timestamp,
        "cluster": args.cluster,
        "similar_task": args.similar_task,
        "mode": args.mode,
        "steps": args.steps,
        "fps": args.fps,
        "total_frames": len(loop_frames),
        "l2_distance": l2_dist,
        "elite_1": {
            "agent_id": top2[0]["agent_id"],
            "reward": top2[0]["reward"],
            "sequence_length": seq1,
        },
        "elite_2": {
            "agent_id": top2[1]["agent_id"],
            "reward": top2[1]["reward"],
            "sequence_length": seq2,
        },
        "gif_path": gif_path,
    }
    meta_path = os.path.join(args.output_dir,
                             f"interpolation_gif_{timestamp}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  Metadata saved: {meta_path}")
    print(f"\n{'='*70}")
    print("  Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
