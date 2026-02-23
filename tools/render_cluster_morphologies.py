#!/usr/bin/env python3
"""
Render sampled morphologies to PNG images using MuJoCo offscreen rendering.

Reads from results/cluster_samples/cluster_{N}/*.xml
Writes to  results/cluster_renders/cluster_{N}/*.png

Usage:
    MUJOCO_GL=egl python tools/render_cluster_morphologies.py [--num_clusters 20]

If EGL rendering fails, try: MUJOCO_GL=osmesa python tools/render_cluster_morphologies.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metamorph"))

import mujoco
import numpy as np


def render_morphology_xml(xml_path, out_png_path, width=640, height=480):
    """Load a morphology XML and render a single frame to PNG."""
    try:
        from metamorph.envs.modules.agent import create_agent_xml
        xml_str = create_agent_xml(xml_path)

        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = mujoco.Renderer(model, height=height, width=width)

        # Set camera for a nice overview
        renderer.update_scene(data, camera=mujoco.MjvCamera(
            type=mujoco.mjtCamera.mjCAMERA_FREE,
            azimuth=135,
            elevation=-25,
            distance=3.5,
            lookat=np.array([0.0, 0.0, 0.3]),
        ))

        pixels = renderer.render()
        renderer.close()

        # Save as PNG using PIL if available, fallback to raw write
        try:
            from PIL import Image
            img = Image.fromarray(pixels)
            img.save(out_png_path)
        except ImportError:
            # Fallback: save as raw PPM (viewable but larger)
            ppm_path = out_png_path.replace(".png", ".ppm")
            with open(ppm_path, "wb") as f:
                f.write(f"P6\n{width} {height}\n255\n".encode())
                f.write(pixels.tobytes())
            print(f"    (PIL not installed, saved as PPM: {ppm_path})")
            return True

        return True
    except Exception as e:
        print(f"  ERROR rendering {xml_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Render cluster morphologies")
    parser.add_argument("--samples_dir", type=str,
                        default="./results/cluster_samples")
    parser.add_argument("--renders_dir", type=str,
                        default="./results/cluster_renders")
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    success = 0
    fail = 0

    for cluster_label in range(args.num_clusters):
        cluster_in_dir = os.path.join(args.samples_dir, f"cluster_{cluster_label}")
        cluster_out_dir = os.path.join(args.renders_dir, f"cluster_{cluster_label}")

        if not os.path.exists(cluster_in_dir):
            print(f"  Cluster {cluster_label}: no samples directory, skipping")
            continue

        os.makedirs(cluster_out_dir, exist_ok=True)

        xml_files = sorted(f for f in os.listdir(cluster_in_dir) if f.endswith(".xml"))
        for xml_file in xml_files:
            xml_path = os.path.join(cluster_in_dir, xml_file)
            out_name = xml_file.replace(".xml", ".png")
            out_path = os.path.join(cluster_out_dir, out_name)

            if render_morphology_xml(xml_path, out_path, args.width, args.height):
                print(f"  Cluster {cluster_label}: rendered {out_name}")
                success += 1
            else:
                fail += 1

    print(f"\nDone: {success} rendered, {fail} failed")
    print(f"Renders saved to {args.renders_dir}")


if __name__ == "__main__":
    main()
