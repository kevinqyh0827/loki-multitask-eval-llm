#!/usr/bin/env python3
"""
Visualize the best-performing morphology from each (cluster, task) evaluation.

For each completed run, finds the agent with the highest mean reward,
renders its morphology XML, and creates a composite 3x3 grid image:
  rows = tasks (locomotion, obstacle, incline)
  cols = clusters (0, 2, 18)

Each cell shows the morphology render annotated with agent ID and reward.

Usage:
    MUJOCO_GL=egl python tools/visualize_best_performers.py
    # or with osmesa fallback:
    MUJOCO_GL=osmesa python tools/visualize_best_performers.py
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metamorph"))

# MuJoCo XML loading uses relative paths from metamorph/ directory
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_METAMORPH_DIR = os.path.join(_PROJECT_ROOT, "metamorph")
os.chdir(_METAMORPH_DIR)

import mujoco
import numpy as np

# ----- configuration -----
TASKS = [
    ("locomotion", "ft"),
    ("obstacle", "obstacle"),
    ("incline", "incline"),
]
CLUSTERS = [0, 2, 18]
SEED = 3429
BASE_OUTPUT = "output/loki"
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "best_performers")


def run_dir(env_type, cluster):
    return os.path.join(
        BASE_OUTPUT, env_type, "kmeans_cluster", "20", str(cluster),
        "walker20", "freq2", "drop2", f"seed{SEED}",
    )


def find_best_from_results_json(rpath):
    """Find best agent from Unimal-v0_results.json (handles truncated files)."""
    with open(rpath) as f:
        content = f.read()

    data = None
    # Try loading directly
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to fix truncated JSON by adding closing braces
        for suffix in ["]}}", "]}}}", "]}]}}}", ']}}}']:
            try:
                data = json.loads(content + suffix)
                break
            except json.JSONDecodeError:
                continue

    if data is None:
        return None, None

    best_agent = None
    best_reward = -float("inf")
    for agent_id, agent_data in data.items():
        if not isinstance(agent_data, dict):
            continue
        rewards = agent_data.get("reward", {}).get("reward", [])
        if len(rewards) > 0:
            mean_r = np.mean(rewards[-10:])
            if mean_r > best_reward:
                best_reward = mean_r
                best_agent = agent_id
    return best_agent, best_reward


def find_best_from_log(task_name, cluster):
    """Fallback: parse reward stats from training log."""
    # Check both old and new log locations
    log_paths = [
        os.path.join(_PROJECT_ROOT, f"log/train_loki_task/{task_name}/cluster20_idx{cluster}_walker20_freq2_drop2_seed{SEED}.log"),
        os.path.join(_PROJECT_ROOT, f"log/train_loki/cluster20_idx{cluster}_walker20_freq2_drop2_seed{SEED}.log"),
    ]
    for log_path in log_paths:
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            content = f.read()
        pattern = r"Agent\s+(\d+):\s+mean/median reward\s+([\d.]+)/([\d.]+)"
        matches = list(re.finditer(pattern, content))
        if len(matches) < 20:
            continue
        last_block = matches[-20:]
        best_agent = None
        best_reward = -float("inf")
        for m in last_block:
            agent_id = m.group(1)
            mean_reward = float(m.group(2))
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_agent = agent_id
        return best_agent, best_reward
    return None, None


def get_best_xml_path(env_type, cluster, agent_id):
    """Find the XML file for the best agent at the last checkpoint iteration."""
    xml_base = os.path.join(run_dir(env_type, cluster), "xml_step")
    if not os.path.isdir(xml_base):
        return None
    iters = sorted(
        [int(d) for d in os.listdir(xml_base) if d.isdigit()]
    )
    if not iters:
        return None
    last_iter = iters[-1]
    xml_path = os.path.join(xml_base, str(last_iter), f"{agent_id}.xml")
    return xml_path if os.path.exists(xml_path) else None


def render_morphology(xml_path, width=800, height=600):
    """Render a morphology XML on a flat terrain with auto-fit camera."""
    from metamorph.envs.modules.agent import create_agent_xml
    import xml.etree.ElementTree as ET

    xml_str = create_agent_xml(xml_path)
    root = ET.fromstring(xml_str)

    # --- Add flat floor to the worldbody ---
    worldbody = root.find("worldbody")
    floor = ET.SubElement(worldbody, "geom")
    floor.set("name", "floor/0")
    floor.set("type", "plane")
    floor.set("pos", "0 0 0")
    floor.set("size", "5 5 0.1")
    floor.set("material", "grid")

    # The agent's torso z-pos is set assuming TERRAIN.SIZE[2]=1 (floor box height).
    # Since we use a plane at z=0, shift the agent down by 1.0 so it sits on the ground.
    torso = worldbody.find(".//body[@name='torso/0']")
    if torso is not None:
        pos = [float(x) for x in torso.get("pos", "0 0 1").split()]
        pos[2] = max(pos[2] - 1.0, 0.05)  # subtract floor box height
        torso.set("pos", " ".join(f"{v:.4f}" for v in pos))

    # --- Inject offscreen framebuffer size ---
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    gl = visual.find("global")
    if gl is None:
        gl = ET.SubElement(visual, "global")
    gl.set("offwidth", str(width))
    gl.set("offheight", str(height))
    xml_str = ET.tostring(root, encoding="unicode")

    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # --- Auto-fit camera to the agent's bounding box ---
    # Compute bounding box of all geom positions
    geom_positions = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and "floor" in name:
            continue
        geom_positions.append(data.geom_xpos[i].copy())
    if geom_positions:
        positions = np.array(geom_positions)
        center = (positions.max(axis=0) + positions.min(axis=0)) / 2
        extent = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
        distance = max(extent * 1.8, 1.5)  # ensure minimum distance
    else:
        center = np.array([0.0, 0.0, 0.5])
        distance = 3.0

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 145
    cam.elevation = -20
    cam.distance = distance
    cam.lookat[:] = center
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    renderer.close()
    return pixels


def add_text_to_image(img, lines, position="bottom"):
    """Add text annotation to image using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    y_offset = img.shape[0] - 25 * len(lines) - 5 if position == "bottom" else 5
    for i, (text, is_title) in enumerate(lines):
        f = font if is_title else font_small
        # Draw text with dark outline for readability
        x = 10
        y = y_offset + i * 25
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=f)
        color = (255, 255, 100) if is_title else (255, 255, 255)
        draw.text((x, y), text, fill=color, font=f)

    return np.array(pil_img)


def create_grid(cell_images, row_labels, col_labels, cell_width=800, cell_height=600):
    """Create a labeled grid image from a 2D array of cell images."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except (OSError, IOError):
        font_label = ImageFont.load_default()

    label_h = 50  # height for column headers
    label_w = 160  # width for row labels
    nrows = len(row_labels)
    ncols = len(col_labels)

    total_w = label_w + ncols * cell_width
    total_h = label_h + nrows * cell_height

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    # Column headers
    for j, col_label in enumerate(col_labels):
        x = label_w + j * cell_width + cell_width // 2
        bbox = draw.textbbox((0, 0), col_label, font=font_label)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 12), col_label, fill=(255, 255, 255), font=font_label)

    # Row labels
    for i, row_label in enumerate(row_labels):
        y = label_h + i * cell_height + cell_height // 2
        bbox = draw.textbbox((0, 0), row_label, font=font_label)
        th = bbox[3] - bbox[1]
        tw = bbox[2] - bbox[0]
        draw.text((label_w // 2 - tw // 2, y - th // 2), row_label, fill=(255, 255, 255), font=font_label)

    # Paste cell images
    for i in range(nrows):
        for j in range(ncols):
            if cell_images[i][j] is not None:
                cell_pil = Image.fromarray(cell_images[i][j])
                # Resize if needed
                if cell_pil.size != (cell_width, cell_height):
                    cell_pil = cell_pil.resize((cell_width, cell_height), Image.LANCZOS)
                canvas.paste(cell_pil, (label_w + j * cell_width, label_h + i * cell_height))
            else:
                # Draw "N/A" for missing cells
                cx = label_w + j * cell_width + cell_width // 2
                cy = label_h + i * cell_height + cell_height // 2
                draw.text((cx - 20, cy - 12), "N/A", fill=(128, 128, 128), font=font_label)

    return np.array(canvas)


def main():
    parser = argparse.ArgumentParser(description="Visualize best performers per (cluster, task)")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cell_images = []
    summary = []

    print("=" * 70)
    print("  Best Performer Visualization: 3 tasks × 3 clusters")
    print("=" * 70)

    for task_name, env_type in TASKS:
        row_images = []
        for cluster in CLUSTERS:
            rd = run_dir(env_type, cluster)
            results_path = os.path.join(rd, "Unimal-v0_results.json")

            # Find best agent
            best_agent, best_reward = None, None
            if os.path.exists(results_path):
                best_agent, best_reward = find_best_from_results_json(results_path)

            # Fallback to log
            if best_agent is None:
                best_agent, best_reward = find_best_from_log(task_name, cluster)

            if best_agent is None:
                print(f"  {task_name}/cluster{cluster}: NO DATA")
                row_images.append(None)
                continue

            # Find XML
            xml_path = get_best_xml_path(env_type, cluster, best_agent)
            if xml_path is None:
                print(f"  {task_name}/cluster{cluster}: agent={best_agent} reward={best_reward:.1f} NO XML")
                row_images.append(None)
                continue

            print(f"  {task_name}/cluster{cluster}: agent={best_agent} reward={best_reward:.1f} -> {xml_path}")
            summary.append({
                "task": task_name, "cluster": cluster,
                "agent": best_agent, "reward": best_reward,
                "xml": xml_path,
            })

            # Render
            try:
                pixels = render_morphology(xml_path, args.width, args.height)
                # Annotate
                pixels = add_text_to_image(pixels, [
                    (f"Agent {best_agent}  |  Reward: {best_reward:.0f}", True),
                    (f"{task_name} / cluster {cluster}", False),
                ])
                row_images.append(pixels)

                # Save individual render
                from PIL import Image
                indiv_path = os.path.join(args.output_dir, f"{task_name}_cluster{cluster}_agent{best_agent}.png")
                Image.fromarray(pixels).save(indiv_path)
            except Exception as e:
                print(f"    ERROR rendering: {e}")
                row_images.append(None)

        cell_images.append(row_images)

    # Create composite grid
    print("\nCreating composite grid...")
    row_labels = [t[0].capitalize() for t in TASKS]
    col_labels = [f"Cluster {c}" for c in CLUSTERS]
    grid = create_grid(cell_images, row_labels, col_labels, args.width, args.height)

    from PIL import Image
    grid_path = os.path.join(args.output_dir, "best_performers_grid.png")
    Image.fromarray(grid).save(grid_path)
    print(f"\nGrid saved to: {grid_path}")

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "best_performers_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Task':<15} {'Cluster 0':<20} {'Cluster 2':<20} {'Cluster 18':<20}")
    print("-" * 70)
    for task_name, env_type in TASKS:
        row = f"{task_name:<15}"
        for c in CLUSTERS:
            match = [s for s in summary if s["task"] == task_name and s["cluster"] == c]
            if match:
                s = match[0]
                row += f"A{s['agent']:>2} R={s['reward']:>7.1f}   "
            else:
                row += f"{'N/A':<20}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    main()
