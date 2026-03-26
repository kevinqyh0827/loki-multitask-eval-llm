#!/usr/bin/env python3
"""Visualize intra-cluster morphology neighbors in VAE latent space.

For each of N randomly selected clusters, picks an anchor morphology,
finds its K nearest neighbors by L2 distance in VAE latent space,
and renders a comparison visualization.

Usage:
    MUJOCO_GL=egl python tools/visualize_cluster_neighbors.py
    MUJOCO_GL=egl python tools/visualize_cluster_neighbors.py --num_clusters 3 --k 5 --seed 42
"""

import argparse
import io
import json
import os
import sys
import tarfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "metamorph"))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from tools.interpolation_demo import (
    load_vae_model,
    vec_to_vae_input,
    encode_to_latent,
    xml_to_latent,
    render_from_xml_string,
    render_morphology_from_xml_path,
    add_text_to_image,
    create_strip_image,
    create_invalid_placeholder,
    get_elite_rewards,
    VAE_CHECKPOINT,
    TASK_DIRS,
    SEED,
)

_METAMORPH_DIR = os.path.join(_PROJECT_ROOT, "metamorph")

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "cluster_neighbors")


def render_from_xml_string_proper(xml_str, width=400, height=300):
    """Render XML string through create_agent_xml for proper pose/materials.

    The raw XML from cluster tar files lacks the UNIMAL base template
    (floor, lighting, materials, proper torso height). This writes to a
    temp file, runs it through create_agent_xml, then renders.
    """
    import tempfile

    saved_cwd = os.getcwd()
    os.chdir(_METAMORPH_DIR)
    from metamorph.envs.modules.agent import create_agent_xml

    with tempfile.NamedTemporaryFile(
        suffix=".xml", mode="w", delete=False, dir="/tmp"
    ) as tmp:
        tmp.write(xml_str)
        tmp_path = tmp.name

    try:
        full_xml = create_agent_xml(tmp_path)
    finally:
        os.unlink(tmp_path)
        os.chdir(saved_cwd)

    return render_from_xml_string(full_xml, width, height)


# =====================================================================
# Cluster data loading
# =====================================================================

def load_cluster_data(tar_path, max_samples=None, seed=None):
    """Load morphology data from a cluster tar file.

    Args:
        tar_path: Path to the cluster tar file.
        max_samples: If set, randomly sample this many morphologies.
        seed: Random seed for sampling.

    Returns:
        List of dicts: [{morph_id, vec [11,14], mask [11,14], xml_content}]
    """
    pt_data = {}   # morph_id -> tensor [11, 28]
    xml_data = {}  # morph_id -> xml string

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.endswith(".pt"):
                morph_id = name[:-3]  # strip ".pt"
                f = tar.extractfile(member)
                if f is not None:
                    tensor = torch.load(
                        io.BytesIO(f.read()), map_location="cpu", weights_only=True
                    )
                    pt_data[morph_id] = tensor
            elif name.endswith(".xml"):
                morph_id = name[:-4]  # strip ".xml"
                f = tar.extractfile(member)
                if f is not None:
                    xml_data[morph_id] = f.read().decode("utf-8")

    # Match .pt and .xml by morph_id
    common_ids = sorted(set(pt_data.keys()) & set(xml_data.keys()))

    if max_samples is not None and max_samples < len(common_ids):
        rng = np.random.RandomState(seed)
        common_ids = list(rng.choice(common_ids, size=max_samples, replace=False))

    data_list = []
    for morph_id in common_ids:
        tensor = pt_data[morph_id].float()  # [11, 28]
        vec = tensor[:, :14]   # [11, 14]
        mask = tensor[:, 14:]  # [11, 14]
        data_list.append({
            "morph_id": morph_id,
            "vec": vec,
            "mask": mask,
            "xml_content": xml_data[morph_id],
        })

    return data_list


# =====================================================================
# Batch VAE encoding
# =====================================================================

def encode_batch_to_latent(data_list, model, device, batch_size=64):
    """Encode all morphologies through the VAE encoder in batches.

    Args:
        data_list: List of dicts from load_cluster_data.
        model: Loaded Model_VAE.
        device: torch device.
        batch_size: Batch size for encoding.

    Returns:
        latent_matrix: [N, 352] tensor (11 * 32 = 352 flattened latent).
    """
    # Prepare VAE inputs for all morphologies
    x_num_list = []
    x_depth_list = []
    for item in data_list:
        x_num, x_depth = vec_to_vae_input(item["vec"])
        x_num_list.append(x_num)      # [11, input_dim]
        x_depth_list.append(x_depth)   # [11, 1]

    all_latents = []
    n = len(x_num_list)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_num_batch = torch.stack(x_num_list[start:end]).to(device)    # [B, 11, input_dim]
        x_depth_batch = torch.stack(x_depth_list[start:end]).to(device)  # [B, 11, 1]

        with torch.no_grad():
            x = model.VAE.Tokenizer(x_num_batch, x_depth_batch)  # [B, 11, H]
            x = x.permute(1, 0, 2)   # [11, B, H]
            x = model.VAE.pos_embedding(x)
            x = x.permute(1, 0, 2)   # [B, 11, H]
            mu_z = model.VAE.encoder_mu(x)  # [B, 11, 32]

        latents = mu_z.reshape(end - start, -1).cpu()  # [B, 352]
        all_latents.append(latents)

        if (start // batch_size) % 10 == 0:
            print(f"    Encoded {end}/{n} morphologies...")

    latent_matrix = torch.cat(all_latents, dim=0)  # [N, 352]
    return latent_matrix


# =====================================================================
# Nearest neighbor search
# =====================================================================

def find_nearest_neighbors(latent_matrix, anchor_idx, k=5):
    """Find k nearest neighbors by L2 distance in latent space.

    Args:
        latent_matrix: [N, D] tensor.
        anchor_idx: Index of the anchor morphology.
        k: Number of nearest neighbors to find.

    Returns:
        (indices, distances): Sorted by distance (excluding self).
    """
    anchor = latent_matrix[anchor_idx].unsqueeze(0)  # [1, D]
    dists = torch.norm(latent_matrix - anchor, dim=1)  # [N]
    # Set self distance to infinity to exclude it
    dists[anchor_idx] = float("inf")
    # Get k nearest
    k_actual = min(k, len(dists) - 1)
    topk_dists, topk_indices = torch.topk(dists, k_actual, largest=False)
    return topk_indices.tolist(), topk_dists.tolist()


# =====================================================================
# Top-rewarded anchor selection
# =====================================================================

def get_top_rewarded_anchors(cluster_id, tasks, eval_data_root, unique_anchors=True):
    """Find the top-rewarded agent per task from eval results.

    Args:
        cluster_id: Cluster index.
        tasks: List of task names (e.g. ["ft", "incline", "obstacle"]).
        eval_data_root: Path to eval_record_details directory.
        unique_anchors: If True, ensure distinct agents across tasks.

    Returns:
        List of dicts: [{task, agent_id, reward, xml_path}]
    """
    task_dir_map = {"ft": "ft", "incline": "incline", "obstacle": "obstacle"}
    eval_data_root = os.path.abspath(eval_data_root)

    # For each task, get sorted list of (agent_id, reward)
    task_agents = {}
    for task in tasks:
        task_dir = task_dir_map.get(task, task)
        rewards = get_elite_rewards(cluster_id, task_dir, data_root=eval_data_root)
        if not rewards:
            print(f"  WARNING: No rewards found for cluster {cluster_id}, task {task}")
            continue
        sorted_agents = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
        task_agents[task] = sorted_agents

    # Select top agent per task with uniqueness constraint
    anchors = []
    used_ids = set()

    for task in tasks:
        if task not in task_agents:
            continue
        for agent_id, reward in task_agents[task]:
            if unique_anchors and agent_id in used_ids:
                continue
            # Build xml path
            task_dir = task_dir_map.get(task, task)
            xml_path = os.path.join(
                eval_data_root, task_dir, "kmeans_cluster", "20",
                str(cluster_id), "walker20", "freq2", "drop2", f"seed{SEED}",
                "xml_step", "1218", f"{agent_id}.xml",
            )
            if os.path.exists(xml_path):
                anchors.append({
                    "task": task,
                    "agent_id": agent_id,
                    "reward": reward,
                    "xml_path": xml_path,
                })
                used_ids.add(agent_id)
                break
            else:
                print(f"  WARNING: XML not found for agent {agent_id} at {xml_path}")

    return anchors


def find_neighbors_for_anchor_latent(anchor_z, latent_matrix, k=5):
    """Find k nearest neighbors to an anchor latent in a cluster pool.

    Args:
        anchor_z: [1, 11, 32] anchor latent tensor.
        latent_matrix: [N, 352] cluster pool latent tensor.
        k: Number of neighbors.

    Returns:
        (indices, distances): Sorted by distance.
    """
    anchor_flat = anchor_z.reshape(1, -1)  # [1, 352]
    dists = torch.norm(latent_matrix - anchor_flat, dim=1)  # [N]
    k_actual = min(k, len(dists))
    topk_dists, topk_indices = torch.topk(dists, k_actual, largest=False)
    return topk_indices.tolist(), topk_dists.tolist()


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize intra-cluster morphology neighbors in VAE latent space"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=3,
        help="Number of random clusters to visualize (default: 3)"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of nearest neighbors per anchor (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num_total_clusters", type=int, default=20,
        help="Total number of clusters (default: 20)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Max morphologies to load per cluster (default: all)"
    )
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help="Output directory for results"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--anchor_mode", type=str, default="random",
        choices=["random", "top_rewarded"],
        help="Anchor selection mode: 'random' (default) or 'top_rewarded'"
    )
    parser.add_argument(
        "--cluster_id", type=int, default=None,
        help="Specific cluster to analyze (required for top_rewarded mode)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["ft", "incline", "obstacle"],
        help="Tasks for top_rewarded mode (default: ft incline obstacle)"
    )
    parser.add_argument(
        "--eval_data_root", type=str, default=None,
        help="Path to eval_record_details directory (for top_rewarded mode)"
    )
    parser.add_argument(
        "--unique_anchors", action="store_true", default=True,
        help="Ensure unique anchor agents across tasks (default: True)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.RandomState(args.seed)

    print("=" * 70)
    print("  Intra-Cluster Nearest Neighbor Visualization")
    print("=" * 70)

    # ---- Step 1: Load VAE ----
    print(f"\nLoading VAE model from {VAE_CHECKPOINT}...")
    model = load_vae_model(VAE_CHECKPOINT, device)
    print("  VAE loaded.")

    # ---- Step 2: Determine mode and load cluster pool ----
    data_dir = os.path.join(_PROJECT_ROOT, "data")
    from PIL import Image

    all_strips = []
    metadata = {
        "seed": args.seed,
        "k": args.k,
        "anchor_mode": args.anchor_mode,
        "clusters": [],
    }

    if args.anchor_mode == "top_rewarded":
        # --- TOP-REWARDED MODE ---
        if args.cluster_id is None:
            print("ERROR: --cluster_id is required for top_rewarded mode.")
            sys.exit(1)
        if args.eval_data_root is None:
            print("ERROR: --eval_data_root is required for top_rewarded mode.")
            sys.exit(1)

        cluster_id = args.cluster_id
        tar_path = os.path.join(
            data_dir, f"latent_cluster{args.num_total_clusters}_{cluster_id}.tar"
        )
        if not os.path.isfile(tar_path):
            print(f"ERROR: Cluster tar file not found: {tar_path}")
            sys.exit(1)

        print(f"\n  Mode: top_rewarded")
        print(f"  Cluster: {cluster_id}")
        print(f"  Tasks: {args.tasks}")

        # Find top-rewarded anchors
        print(f"\nFinding top-rewarded anchors...")
        anchors = get_top_rewarded_anchors(
            cluster_id, args.tasks, args.eval_data_root,
            unique_anchors=args.unique_anchors,
        )
        print(f"  Found {len(anchors)} anchors:")
        for a in anchors:
            print(f"    {a['task']}: Agent {a['agent_id']} (R={a['reward']:.1f})")

        # Load cluster pool
        print(f"\nLoading cluster {cluster_id} pool from tar...")
        data_list = load_cluster_data(
            tar_path, max_samples=args.max_samples, seed=args.seed
        )
        print(f"  Loaded {len(data_list)} morphologies.")

        # Encode cluster pool to latent space
        print(f"  Encoding cluster pool to VAE latent space...")
        latent_matrix = encode_batch_to_latent(data_list, model, device)
        print(f"  Latent matrix shape: {latent_matrix.shape}")

        metadata["cluster_id"] = cluster_id
        metadata["tasks"] = args.tasks
        metadata["anchors"] = []

        # Process each anchor
        for anchor_info in anchors:
            task = anchor_info["task"]
            print(f"\n{'=' * 70}")
            print(f"  Task: {task} | Anchor: Agent {anchor_info['agent_id']} (R={anchor_info['reward']:.1f})")
            print(f"{'=' * 70}")

            # Encode anchor through VAE
            print(f"  Encoding anchor...")
            anchor_z, anchor_vec = xml_to_latent(anchor_info["xml_path"], model, device)

            # Find neighbors in cluster pool
            nn_indices, nn_distances = find_neighbors_for_anchor_latent(
                anchor_z, latent_matrix, k=args.k
            )
            print(f"  Nearest neighbors (L2 distances):")
            for rank, (idx, dist) in enumerate(zip(nn_indices, nn_distances), 1):
                print(f"    #{rank}: morph_id={data_list[idx]['morph_id']}, dist={dist:.4f}")

            # Render anchor + neighbors
            print(f"  Rendering morphologies...")
            images = []
            labels = []

            # Render anchor from XML path
            try:
                px = render_morphology_from_xml_path(
                    anchor_info["xml_path"], args.width, args.height
                )
                px = add_text_to_image(px, [
                    f"ANCHOR: Agent {anchor_info['agent_id']}",
                    f"R={anchor_info['reward']:.0f} ({task})",
                ])
                images.append(px)
            except Exception as e:
                print(f"    Anchor render failed: {e}")
                images.append(create_invalid_placeholder(args.width, args.height))
            labels.append(f"ANCHOR Ag.{anchor_info['agent_id']} (R={anchor_info['reward']:.0f})")

            # Render neighbors
            for rank, (idx, dist) in enumerate(zip(nn_indices, nn_distances), 1):
                neighbor = data_list[idx]
                try:
                    px = render_from_xml_string_proper(
                        neighbor["xml_content"], args.width, args.height
                    )
                    px = add_text_to_image(px, [
                        f"#{rank}: {neighbor['morph_id']}",
                        f"dist={dist:.4f}",
                    ])
                    images.append(px)
                except Exception as e:
                    print(f"    Neighbor #{rank} render failed: {e}")
                    images.append(create_invalid_placeholder(args.width, args.height))
                labels.append(f"#{rank} {neighbor['morph_id']} (d={dist:.3f})")

            # Create strip
            strip = create_strip_image(
                images, labels,
                strip_label=(
                    f"Cluster {cluster_id} | Task: {task} | "
                    f"Anchor: Agent {anchor_info['agent_id']} (R={anchor_info['reward']:.0f}) "
                    f"+ {len(nn_indices)} Nearest Neighbors"
                ),
                cell_width=args.width, cell_height=args.height,
            )
            all_strips.append(strip)

            # Record metadata
            metadata["anchors"].append({
                "task": task,
                "agent_id": anchor_info["agent_id"],
                "reward": anchor_info["reward"],
                "xml_path": anchor_info["xml_path"],
                "neighbors": [
                    {
                        "rank": rank,
                        "morph_id": data_list[idx]["morph_id"],
                        "index": int(idx),
                        "l2_distance": float(dist),
                    }
                    for rank, (idx, dist) in enumerate(zip(nn_indices, nn_distances), 1)
                ],
            })

    else:
        # --- RANDOM MODE (original behavior) ---
        available_clusters = []
        for i in range(args.num_total_clusters):
            tar_path = os.path.join(
                data_dir, f"latent_cluster{args.num_total_clusters}_{i}.tar"
            )
            if os.path.isfile(tar_path):
                available_clusters.append(i)

        if not available_clusters:
            print(f"ERROR: No cluster tar files found in {data_dir}")
            print(f"  Expected pattern: latent_cluster{args.num_total_clusters}_<i>.tar")
            sys.exit(1)

        print(f"\nFound {len(available_clusters)} cluster tar files.")

        num_to_select = min(args.num_clusters, len(available_clusters))
        selected_clusters = sorted(
            rng.choice(available_clusters, size=num_to_select, replace=False).tolist()
        )
        print(f"Selected clusters: {selected_clusters}")
        metadata["selected_clusters"] = selected_clusters

        for cluster_idx in selected_clusters:
            tar_path = os.path.join(
                data_dir, f"latent_cluster{args.num_total_clusters}_{cluster_idx}.tar"
            )
            print(f"\n{'=' * 70}")
            print(f"  Cluster {cluster_idx}")
            print(f"{'=' * 70}")

            print(f"  Loading data from {os.path.basename(tar_path)}...")
            data_list = load_cluster_data(
                tar_path, max_samples=args.max_samples, seed=args.seed
            )
            print(f"  Loaded {len(data_list)} morphologies.")

            if len(data_list) < args.k + 1:
                print(f"  WARNING: Only {len(data_list)} morphologies, need at least "
                      f"{args.k + 1}. Skipping cluster {cluster_idx}.")
                continue

            print(f"  Encoding to VAE latent space...")
            latent_matrix = encode_batch_to_latent(data_list, model, device)
            print(f"  Latent matrix shape: {latent_matrix.shape}")

            anchor_idx = rng.randint(0, len(data_list))
            anchor = data_list[anchor_idx]
            print(f"  Anchor: morph_id={anchor['morph_id']} (index {anchor_idx})")

            nn_indices, nn_distances = find_nearest_neighbors(
                latent_matrix, anchor_idx, k=args.k
            )
            print(f"  Nearest neighbors (L2 distances):")
            for rank, (idx, dist) in enumerate(zip(nn_indices, nn_distances), 1):
                print(f"    #{rank}: morph_id={data_list[idx]['morph_id']}, dist={dist:.4f}")

            print(f"  Rendering morphologies...")
            images = []
            labels = []

            try:
                px = render_from_xml_string_proper(
                    anchor["xml_content"], args.width, args.height
                )
                px = add_text_to_image(px, [
                    f"ANCHOR: {anchor['morph_id']}",
                    "dist=0.0000",
                ])
                images.append(px)
            except Exception as e:
                print(f"    Anchor render failed: {e}")
                images.append(create_invalid_placeholder(args.width, args.height))
            labels.append(f"ANCHOR {anchor['morph_id']} (d=0.0)")

            for rank, (idx, dist) in enumerate(zip(nn_indices, nn_distances), 1):
                neighbor = data_list[idx]
                try:
                    px = render_from_xml_string_proper(
                        neighbor["xml_content"], args.width, args.height
                    )
                    px = add_text_to_image(px, [
                        f"#{rank}: {neighbor['morph_id']}",
                        f"dist={dist:.4f}",
                    ])
                    images.append(px)
                except Exception as e:
                    print(f"    Neighbor #{rank} render failed: {e}")
                    images.append(create_invalid_placeholder(args.width, args.height))
                labels.append(f"#{rank} {neighbor['morph_id']} (d={dist:.3f})")

            strip = create_strip_image(
                images, labels,
                strip_label=(
                    f"Cluster {cluster_idx}: Anchor {anchor['morph_id']} "
                    f"+ {len(nn_indices)} Nearest Neighbors"
                ),
                cell_width=args.width, cell_height=args.height,
            )
            all_strips.append(strip)

            cluster_meta = {
                "cluster_idx": cluster_idx,
                "num_morphologies": len(data_list),
                "anchor_morph_id": anchor["morph_id"],
                "anchor_index": int(anchor_idx),
                "neighbors": [
                    {
                        "rank": rank,
                        "morph_id": data_list[idx]["morph_id"],
                        "index": int(idx),
                        "l2_distance": float(dist),
                    }
                    for rank, (idx, dist) in enumerate(zip(nn_indices, nn_distances), 1)
                ],
            }
            metadata["clusters"].append(cluster_meta)

    # ---- Step 4: Assemble composite image ----
    if not all_strips:
        print("\nERROR: No strips were generated. Check cluster data.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("  Assembling composite visualization")
    print(f"{'=' * 70}")

    # Stack rows vertically, padding to same width
    max_w = max(s.shape[1] for s in all_strips)
    padded = []
    for s in all_strips:
        if s.shape[1] < max_w:
            pad = np.full(
                (s.shape[0], max_w - s.shape[1], 3), 40, dtype=np.uint8
            )
            s = np.concatenate([s, pad], axis=1)
        padded.append(s)
    composite = np.concatenate(padded, axis=0)

    # Save image
    if args.anchor_mode == "top_rewarded":
        cluster_str = str(args.cluster_id)
        task_str = "_".join(args.tasks)
        fname_base = f"cluster_neighbors_k{args.k}_c{cluster_str}_{task_str}"
    else:
        cluster_str = "_".join(str(c) for c in selected_clusters)
        fname_base = f"cluster_neighbors_k{args.k}_c{cluster_str}_s{args.seed}"

    out_path = os.path.join(args.output_dir, f"{fname_base}.png")
    Image.fromarray(composite).save(out_path)
    print(f"  Visualization saved: {out_path}")

    # Save metadata
    meta_path = os.path.join(args.output_dir, f"{fname_base}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved: {meta_path}")

    print(f"\n{'=' * 70}")
    print("  Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
