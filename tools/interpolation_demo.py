#!/usr/bin/env python3
"""Elite Pool Latent Space Sampling Demo.

Demonstrates the full LOKI pipeline for a new unseen task:
  1. LLM recommends the best cluster based on cross-task performance analysis
  2. LLM identifies the most similar known task
  3. Load elite agents from (best_cluster, most_similar_task)
  4. Encode them into VAE latent space
  5. Generate candidate morphologies via:
     - Mean latent (centroid of elite pool)
     - Reward-weighted mean (emphasize top performers)
  6. Decode, render, and visualize results

Usage:
    MUJOCO_GL=egl python tools/interpolation_demo.py --new_task bump
    MUJOCO_GL=egl python tools/interpolation_demo.py --new_task bump --cluster 18 --similar_task obstacle
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.nn.functional import one_hot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metamorph"))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_METAMORPH_DIR = os.path.join(_PROJECT_ROOT, "metamorph")

from vae.model import Model_VAE
from tools.vec_to_morphology import postprocess_vae_output, vec_to_xml_safe
from tools.util import (
    CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN, D_TOKEN, N_TOKENS,
    THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS,
)

# VAE hyperparameters
NUM_LAYERS = 4
H_DIM = 32
D_DEPTH = 32
N_HEAD = 4
FACTOR = 8

# Paths
VAE_CHECKPOINT = os.path.join(
    _PROJECT_ROOT, "vae", "checkpoints",
    "VAE_50k_hdim32_depth32_LR_0.0001_WD_1e-05_L4_H4_F8_beta0.01_bsize4096_epochs200_20260209_215618",
    "model.pt",
)
BEST_PERFORMERS_JSON = os.path.join(
    _PROJECT_ROOT, "results", "best_performers", "best_performers_summary.json"
)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "interpolation")

# Training output structure
SEED = 3429
TASK_DIRS = {
    "locomotion": "ft",
    "obstacle": "obstacle",
    "incline": "incline",
}
CLUSTERS = [0, 2, 18]


# =====================================================================
# VAE I/O
# =====================================================================

def load_vae_model(checkpoint_path, device):
    """Load pretrained Model_VAE."""
    model = Model_VAE(
        num_layers=NUM_LAYERS, n_tokens=N_TOKENS, d_token=D_TOKEN,
        d_depth=D_DEPTH, hid_dim=H_DIM, n_head=N_HEAD, factor=FACTOR,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def vec_to_vae_input(vec):
    """Convert a [11, 14] raw vec to VAE input tensors (x_num, x_depth)."""
    vec = vec.float().clone()
    vec[vec < 0] = 0
    x_con, x_cat, x_binary, x_depth = vec.split(
        [CONTINUOUS_TOKEN, CATEGORY_TOKEN, BINARY_TOKEN, 1], dim=-1
    )
    num_classes_list = [THETA_CLASS, PHI_CLASS, JOINTX_CLASS, JOINTY_CLASS]
    one_hot_cats = torch.cat([
        one_hot(x_cat[:, i].long(), num_classes=num_classes_list[i])
        for i in range(CATEGORY_TOKEN)
    ], dim=-1).float()
    x_num = torch.cat([x_con, one_hot_cats, x_binary], dim=-1)
    return x_num, x_depth


def encode_to_latent(model, x_num, x_depth, device):
    """Encode through VAE with proper positional encoding."""
    x_num = x_num.unsqueeze(0).to(device)
    x_depth = x_depth.unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.VAE.Tokenizer(x_num, x_depth)
        x = x.permute(1, 0, 2)
        x = model.VAE.pos_embedding(x)
        x = x.permute(1, 0, 2)
        mu_z = model.VAE.encoder_mu(x)
    return mu_z  # [1, 11, 32]


def decode_from_latent(model, z, device):
    """Decode a latent code to a [1, 11, 14] vec."""
    z = z.to(device)
    with torch.no_grad():
        h = model.VAE.decoder(z)
        recon_x_num, recon_x_cat, recon_x_depth = model.Reconstructor(h)
        vec = postprocess_vae_output(recon_x_num, recon_x_cat, recon_x_depth)
    return vec


def xml_to_latent(xml_path, model, device):
    """Full forward pipeline: XML -> pkl -> vec -> encode -> z."""
    saved_cwd = os.getcwd()
    os.chdir(_METAMORPH_DIR)
    from tools.xml_2_pkl import xml_to_pkl
    from tools.pkl_2_vec_new import pkl_to_vec
    pkl = xml_to_pkl(xml_path)
    vec, mask = pkl_to_vec(pkl)
    os.chdir(saved_cwd)
    x_num, x_depth = vec_to_vae_input(vec)
    mu_z = encode_to_latent(model, x_num, x_depth, device)
    return mu_z, vec


# =====================================================================
# Elite pool loading
# =====================================================================

def get_elite_xml_paths(cluster, task_dir):
    """Get XML paths for all elite agents in a cluster/task."""
    xml_dir = os.path.join(
        _METAMORPH_DIR, "output", "loki", task_dir, "kmeans_cluster", "20",
        str(cluster), "walker20", "freq2", "drop2", f"seed{SEED}",
        "xml_step", "1218",
    )
    if not os.path.isdir(xml_dir):
        return []
    return [
        os.path.join(xml_dir, f)
        for f in sorted(os.listdir(xml_dir))
        if f.endswith(".xml") and not f.startswith("tmp_")
    ]


def get_elite_rewards(cluster, task_dir):
    """Extract final rewards per agent from training logs."""
    results_path = os.path.join(
        _METAMORPH_DIR, "output", "loki", task_dir, "kmeans_cluster", "20",
        str(cluster), "walker20", "freq2", "drop2", f"seed{SEED}",
        "Unimal-v0_results.json",
    )
    if not os.path.isfile(results_path):
        return {}
    with open(results_path) as f:
        text = f.read()
    # Extract reward arrays via regex (files may be truncated)
    pattern = r'"(\d+)":\s*\{\s*"reward":\s*\{\s*"reward":\s*\[([\d.,\s\n]+?)\]'
    matches = re.findall(pattern, text)
    rewards = {}
    for agent_id, vals_str in matches:
        vals = [float(v.strip()) for v in vals_str.split(",") if v.strip()]
        if vals:
            rewards[agent_id] = vals[-1]
    return rewards


def load_task_elite_pool(cluster, task_name, model, device):
    """Load elite agents from a single (cluster, task) pair.

    Returns list of dicts: {agent_id, task, xml_path, z, vec, reward}
    """
    task_dir = TASK_DIRS[task_name]
    xml_paths = get_elite_xml_paths(cluster, task_dir)
    rewards = get_elite_rewards(cluster, task_dir)

    pool = []
    for xml_path in xml_paths:
        agent_id = os.path.basename(xml_path).replace(".xml", "")
        reward = rewards.get(agent_id, 0.0)
        try:
            mu_z, vec = xml_to_latent(xml_path, model, device)
            pool.append({
                "agent_id": agent_id,
                "task": task_name,
                "xml_path": xml_path,
                "z": mu_z,  # [1, 11, 32]
                "vec": vec,
                "reward": reward,
            })
        except Exception as e:
            print(f"    SKIP {task_name}/{agent_id}: {e}")
    return pool


# =====================================================================
# Sampling strategies
# =====================================================================

def mean_latent(pool):
    """Compute the centroid of all elite latent codes."""
    z_all = torch.cat([a["z"] for a in pool], dim=0)  # [N, 11, 32]
    return z_all.mean(dim=0, keepdim=True)  # [1, 11, 32]


def reward_weighted_mean(pool, temperature=1.0):
    """Reward-weighted average of elite latent codes.

    Higher rewards get more weight. Temperature controls sharpness:
      - temperature -> 0: winner-take-all (best agent only)
      - temperature -> inf: uniform average
    """
    rewards = np.array([a["reward"] for a in pool])
    # Normalize to avoid overflow
    rewards_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
    weights = np.exp(rewards_norm / temperature)
    weights = weights / weights.sum()

    z_all = torch.cat([a["z"] for a in pool], dim=0)  # [N, 11, 32]
    weights_t = torch.tensor(weights, dtype=z_all.dtype).to(z_all.device)
    z_weighted = (z_all * weights_t[:, None, None]).sum(dim=0, keepdim=True)
    return z_weighted  # [1, 11, 32]


# =====================================================================
# Rendering
# =====================================================================

def render_from_xml_string(xml_str, width=400, height=300):
    """Render a morphology from XML string using MuJoCo offscreen."""
    import mujoco

    root = ET.fromstring(xml_str)
    worldbody = root.find("worldbody")

    floor = ET.SubElement(worldbody, "geom")
    floor.set("name", "floor/0")
    floor.set("type", "plane")
    floor.set("pos", "0 0 0")
    floor.set("size", "5 5 0.1")
    floor.set("material", "grid")

    torso = worldbody.find(".//body[@name='torso/0']")
    if torso is not None:
        pos = [float(x) for x in torso.get("pos", "0 0 1").split()]
        pos[2] = max(pos[2] - 1.0, 0.05)
        torso.set("pos", " ".join(f"{v:.4f}" for v in pos))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    gl = visual.find("global")
    if gl is None:
        gl = ET.SubElement(visual, "global")
    gl.set("offwidth", str(width))
    gl.set("offheight", str(height))

    xml_out = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_out)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

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
        distance = max(extent * 1.8, 1.5)
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


def render_morphology_from_xml_path(xml_path, width=400, height=300):
    """Render using create_agent_xml (for original elite agents)."""
    saved_cwd = os.getcwd()
    os.chdir(_METAMORPH_DIR)
    from metamorph.envs.modules.agent import create_agent_xml
    xml_str = create_agent_xml(xml_path)
    os.chdir(saved_cwd)
    return render_from_xml_string(xml_str, width, height)


def add_text_to_image(img, lines, position="bottom"):
    """Add text annotation to image."""
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    y_offset = img.shape[0] - 20 * len(lines) - 5 if position == "bottom" else 5
    for i, text in enumerate(lines):
        x, y = 5, y_offset + i * 20
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(pil_img)


def create_invalid_placeholder(width=400, height=300):
    """Create a gray placeholder image for invalid morphologies."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (width, height), (80, 80, 80))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((width // 2 - 40, height // 2 - 10), "Invalid", fill=(200, 50, 50), font=font)
    return np.array(img)


def create_strip_image(images, labels, strip_label="", cell_width=400, cell_height=300):
    """Create a horizontal strip from a list of images with labels."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    n = len(images)
    label_h = 40 if strip_label else 0
    sublabel_h = 25
    total_w = n * cell_width
    total_h = label_h + cell_height + sublabel_h

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    if strip_label:
        draw.text((10, 10), strip_label, fill=(255, 255, 100), font=font)

    for i, (img, label) in enumerate(zip(images, labels)):
        cell_pil = Image.fromarray(img)
        if cell_pil.size != (cell_width, cell_height):
            cell_pil = cell_pil.resize((cell_width, cell_height), Image.LANCZOS)
        canvas.paste(cell_pil, (i * cell_width, label_h))

        bbox = draw.textbbox((0, 0), label, font=font_small)
        tw = bbox[2] - bbox[0]
        x = i * cell_width + (cell_width - tw) // 2
        draw.text((x, label_h + cell_height + 3), label, fill=(200, 200, 200), font=font_small)

    return np.array(canvas)


# =====================================================================
# Decode + render helper
# =====================================================================

def decode_render(model, z, device, width, height, label_lines=None):
    """Decode latent -> vec -> XML -> render. Returns (image, xml_str) or (placeholder, None)."""
    decoded_vec = decode_from_latent(model, z, device)
    xml_str = vec_to_xml_safe(decoded_vec[0])
    if xml_str is not None:
        try:
            pixels = render_from_xml_string(xml_str, width, height)
            if label_lines:
                pixels = add_text_to_image(pixels, label_lines)
            return pixels, xml_str
        except Exception as e:
            print(f"    Render failed: {e}")
    return create_invalid_placeholder(width, height), None


# =====================================================================
# LLM recommendation loading
# =====================================================================

def load_recommendation(new_task, project_root):
    """Load the most recent LLM recommendation for a task.

    Returns (best_cluster, most_similar_task, recommendation_data) or raises.
    """
    rec_dir = os.path.join(project_root, "results", "llm_recommendations")
    rec_files = sorted([
        f for f in os.listdir(rec_dir)
        if f.startswith(f"recommendation_{new_task}_") and f.endswith(".json")
    ])
    if not rec_files:
        raise FileNotFoundError(
            f"No LLM recommendation found for '{new_task}'. "
            f"Run: python tools/llm_cluster_recommender.py --new_task {new_task}"
        )
    rec_path = os.path.join(rec_dir, rec_files[-1])
    with open(rec_path) as f:
        rec_data = json.load(f)

    parsed = rec_data.get("parsed_recommendation", {})
    recs = sorted(parsed.get("recommendations", []),
                   key=lambda r: r.get("predicted_rank", 99))
    best_cluster = recs[0]["cluster_id"]

    # Extract most similar task from recommendation
    most_similar = parsed.get("most_similar_task", None)

    # Fallback: check task_similarity scores if available
    if not most_similar and "task_similarity" in parsed:
        ts_list = parsed["task_similarity"]
        best_ts = max(ts_list, key=lambda t: t.get("overall_similarity", 0))
        most_similar = best_ts.get("known_task")

    # Fallback: infer from key_transfer_insights text
    if not most_similar:
        insights = parsed.get("key_transfer_insights", "")
        analysis = parsed.get("new_task_analysis", "")
        combined = (insights + " " + analysis).lower()
        # Score each known task by mention frequency in transfer reasoning
        task_scores = {}
        for task in TASK_DIRS:
            task_scores[task] = combined.count(task)
        if task_scores:
            most_similar = max(task_scores, key=task_scores.get)

    if not most_similar:
        most_similar = "obstacle"  # safe default

    return best_cluster, most_similar, rec_data


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Elite Pool Latent Space Sampling Demo")
    parser.add_argument("--new_task", type=str, default="bump",
                        help="New task name (must have LLM recommendation)")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Override: use this cluster instead of LLM recommendation")
    parser.add_argument("--similar_task", type=str, default=None,
                        choices=list(TASK_DIRS.keys()),
                        help="Override: use this as the most similar known task")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("  Elite Pool Latent Space Sampling Demo")
    print("=" * 70)

    # ---- Step 1: Determine best cluster and most similar task ----
    if args.cluster is not None and args.similar_task is not None:
        best_cluster = args.cluster
        most_similar_task = args.similar_task
        print(f"\nUsing manual overrides:")
        print(f"  Best cluster: {best_cluster}")
        print(f"  Most similar task: {most_similar_task}")
        rec_data = None
    else:
        best_cluster_rec, similar_task_rec, rec_data = load_recommendation(
            args.new_task, _PROJECT_ROOT
        )
        best_cluster = args.cluster if args.cluster is not None else best_cluster_rec
        most_similar_task = args.similar_task if args.similar_task is not None else similar_task_rec

        parsed = rec_data.get("parsed_recommendation", {})
        recs = sorted(parsed.get("recommendations", []),
                       key=lambda r: r.get("predicted_rank", 99))
        print(f"\nLLM Recommendation (task: {args.new_task}):")
        for r in recs:
            marker = " <-- selected" if r["cluster_id"] == best_cluster else ""
            print(f"  #{r['predicted_rank']}. Cluster {r['cluster_id']}: "
                  f"{r['predicted_performance']} ({r['confidence']} confidence){marker}")
        print(f"\n  Best cluster: {best_cluster}")
        print(f"  Most similar known task: {most_similar_task}")

    # ---- Step 2: Load VAE ----
    print(f"\nLoading VAE model...")
    model = load_vae_model(VAE_CHECKPOINT, device)
    print("  VAE loaded")

    # ---- Step 3: Load elite pool from (best_cluster, most_similar_task) ----
    print(f"\nLoading elite pool: Cluster {best_cluster} x {most_similar_task}...")
    pool = load_task_elite_pool(best_cluster, most_similar_task, model, device)
    print(f"  Encoded {len(pool)} elite agents")

    # Summarize rewards
    rewards_available = sum(1 for a in pool if a["reward"] > 0)
    if rewards_available > 0:
        valid_rewards = [a["reward"] for a in pool if a["reward"] > 0]
        print(f"  Rewards available: {rewards_available}/{len(pool)}")
        print(f"  Reward range: [{min(valid_rewards):.0f}, {max(valid_rewards):.0f}]")
    else:
        print(f"  WARNING: No reward data available (truncated log file)")
        print(f"  Reward-weighted sampling will fall back to uniform weights")

    # ---- Step 4: Generate candidate morphologies ----
    print("\n" + "=" * 70)
    print("  Sampling Strategies")
    print("=" * 70)

    from PIL import Image
    metadata = {
        "new_task": args.new_task,
        "best_cluster": best_cluster,
        "most_similar_task": most_similar_task,
        "elite_pool_size": len(pool),
        "rewards_available": rewards_available,
        "strategies": [],
        "validity_stats": {"total": 0, "valid": 0},
    }

    xml_dir = os.path.join(args.output_dir, "xmls")
    os.makedirs(xml_dir, exist_ok=True)

    # --- Strategy 1: Mean Latent (Centroid) ---
    print("\n--- Strategy 1: Mean Latent (Elite Pool Centroid) ---")
    z_mean = mean_latent(pool)
    img_mean, xml_mean = decode_render(
        model, z_mean, device, args.width, args.height,
        ["Mean Latent", f"C{best_cluster} x {most_similar_task}"]
    )
    valid_mean = xml_mean is not None
    metadata["validity_stats"]["total"] += 1
    if valid_mean:
        metadata["validity_stats"]["valid"] += 1
        with open(os.path.join(xml_dir, "mean_latent.xml"), "w") as f:
            f.write(xml_mean)
    metadata["strategies"].append({
        "name": "mean_latent",
        "valid": valid_mean,
        "description": f"Centroid of {len(pool)} elite latent codes from C{best_cluster}/{most_similar_task}",
    })
    print(f"  Result: {'VALID' if valid_mean else 'INVALID'}")

    # --- Strategy 2: Reward-Weighted Mean (multiple temperatures) ---
    print("\n--- Strategy 2: Reward-Weighted Mean ---")
    temperatures = [0.1, 0.3, 0.5, 1.0]
    weighted_images = []
    weighted_labels = []
    for temp in temperatures:
        z_rw = reward_weighted_mean(pool, temperature=temp)
        img_rw, xml_rw = decode_render(
            model, z_rw, device, args.width, args.height,
            [f"Reward-Weighted T={temp}", f"C{best_cluster} x {most_similar_task}"]
        )
        weighted_images.append(img_rw)
        weighted_labels.append(f"T={temp}")
        metadata["validity_stats"]["total"] += 1
        valid = xml_rw is not None
        if valid:
            metadata["validity_stats"]["valid"] += 1
            with open(os.path.join(xml_dir, f"reward_weighted_T{temp}.xml"), "w") as f:
                f.write(xml_rw)
        metadata["strategies"].append({
            "name": f"reward_weighted_T{temp}",
            "valid": valid,
            "temperature": temp,
        })
        print(f"  T={temp}: {'VALID' if valid else 'INVALID'}")

    # ---- Step 5: Render original elite morphologies for comparison ----
    print("\n--- Rendering original elite morphologies for comparison ---")
    num_elites_to_show = min(5, len(pool))
    # Pick top-reward elites (or first N if no rewards)
    sorted_pool = sorted(pool, key=lambda a: a["reward"], reverse=True)
    elites_to_show = sorted_pool[:num_elites_to_show]

    elite_images = []
    elite_labels = []
    for a in elites_to_show:
        try:
            px = render_morphology_from_xml_path(a["xml_path"], args.width, args.height)
            reward_str = f"R={a['reward']:.0f}" if a["reward"] > 0 else "R=?"
            px = add_text_to_image(px, [
                f"Agent {a['agent_id']}", reward_str,
            ])
            elite_images.append(px)
            elite_labels.append(f"Agent {a['agent_id']} ({reward_str})")
        except Exception as e:
            print(f"    Render failed for agent {a['agent_id']}: {e}")
            elite_images.append(create_invalid_placeholder(args.width, args.height))
            elite_labels.append(f"Agent {a['agent_id']} (err)")
    print(f"  Rendered {len(elite_images)} original elites")

    # ---- Step 6: Composite visualization ----
    print("\n--- Assembling visualization ---")
    all_strips = []

    # Row 1: Original elite morphologies
    strip_elites = create_strip_image(
        elite_images, elite_labels,
        strip_label=f"Cluster {best_cluster} x {most_similar_task}: "
                    f"Top-{num_elites_to_show} Original Elite Morphologies",
        cell_width=args.width, cell_height=args.height,
    )
    all_strips.append(strip_elites)

    # Row 2: Mean latent + reward-weighted variants
    sampling_images = [img_mean] + weighted_images
    sampling_labels = ["Mean (uniform)"] + weighted_labels
    strip_sampling = create_strip_image(
        sampling_images, sampling_labels,
        strip_label=f"Cluster {best_cluster} x {most_similar_task} -> {args.new_task}: "
                    f"Centroid & Reward-Weighted Sampling",
        cell_width=args.width, cell_height=args.height,
    )
    all_strips.append(strip_sampling)

    # Stack rows vertically, padding to same width
    max_w = max(s.shape[1] for s in all_strips)
    padded = []
    for s in all_strips:
        if s.shape[1] < max_w:
            pad = np.full((s.shape[0], max_w - s.shape[1], 3), 40, dtype=np.uint8)
            s = np.concatenate([s, pad], axis=1)
        padded.append(s)
    composite = np.concatenate(padded, axis=0)

    out_path = os.path.join(
        args.output_dir,
        f"elite_sampling_C{best_cluster}_{most_similar_task}.png"
    )
    Image.fromarray(composite).save(out_path)
    print(f"  Visualization saved: {out_path}")

    # ---- Step 7: Save metadata ----
    stats = metadata["validity_stats"]
    print(f"\n--- Results ---")
    print(f"  Total generated: {stats['total']}")
    print(f"  Valid morphologies: {stats['valid']}")
    print(f"  Validity rate: {stats['valid']/max(stats['total'],1)*100:.1f}%")

    meta_path = os.path.join(
        args.output_dir,
        f"elite_sampling_C{best_cluster}_{most_similar_task}_metadata.json"
    )
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved: {meta_path}")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
