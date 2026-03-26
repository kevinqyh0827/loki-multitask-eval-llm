#!/usr/bin/env python3
"""End-to-end LLM-guided morphology design demo.

Demonstrates the full LOKI pipeline for a new unseen task:
  1. LLM analyzes task similarity and recommends best (cluster, task) pair
  2. Load elite pool from recommended cluster/task
  3. Deduplicate and select top-2 unique morphologies
  4. Generate new morphologies via midpoint interpolation in both
     VAE latent space and raw design space
  5. Render and visualize all 4 morphologies (2 parents + 2 midpoints)

Default scenario: manipulation_ball as the unseen task, with training data
from locomotion, obstacle, and incline across clusters 0, 2, 18.

Usage:
    # Full pipeline (calls Claude API)
    MUJOCO_GL=egl python tools/end_to_end_demo.py --new_task manipulation_ball

    # Skip LLM step (reuse saved recommendation)
    MUJOCO_GL=egl python tools/end_to_end_demo.py --new_task manipulation_ball --skip_llm

    # Manual cluster/task override
    MUJOCO_GL=egl python tools/end_to_end_demo.py --new_task manipulation_ball --skip_llm --cluster 18 --similar_task obstacle
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup (same pattern as all other tools)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "metamorph"))

# ---------------------------------------------------------------------------
# Imports from existing tools (reuse, not reinvent)
# ---------------------------------------------------------------------------

# LLM recommender
from tools.llm_cluster_recommender import (
    SYSTEM_PROMPT,
    build_user_prompt,
    call_claude,
    extract_json_from_response,
)

# Interpolation demo (VAE, elite pools, rendering)
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
    TASK_DIRS,
)

# Midpoint interpolation (dedup)
from tools.midpoint_interpolation import deduplicate_pool

# Raw-space interpolation
from tools.raw_space_interpolation import interpolate_raw

# Design vector utilities
from tools.xml_to_design_space import get_sequence_length
from tools.vec_to_morphology import vec_to_xml_safe

# Task descriptions (includes manipulation_ball)
from tools.task_similarity_matrix import ALL_TASK_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
_BEST_PERFORMERS_PATH = os.path.join(
    _RESULTS_DIR, "best_performers", "best_performers_summary.json"
)
_REC_OUTPUT_DIR = os.path.join(_RESULTS_DIR, "llm_recommendations")
_DEFAULT_OUTPUT_DIR = os.path.join(_RESULTS_DIR, "end_to_end_demo")


# ---------------------------------------------------------------------------
# Helper: extract recommendation info from parsed JSON
# ---------------------------------------------------------------------------

def extract_recommendation_info(parsed_json):
    """Extract best_cluster and most_similar_task from parsed LLM JSON.

    Uses the same fallback chain as load_recommendation() in
    interpolation_demo.py for consistency.
    """
    # Best cluster: first recommendation sorted by predicted_rank
    recs = sorted(
        parsed_json.get("recommendations", []),
        key=lambda r: r.get("predicted_rank", 99),
    )
    if not recs:
        raise ValueError("No recommendations found in LLM response JSON")
    best_cluster = recs[0]["cluster_id"]

    # Most similar task: try multiple fields
    most_similar = parsed_json.get("most_similar_task", None)

    if not most_similar and "most_similar_tasks" in parsed_json:
        mst_list = parsed_json["most_similar_tasks"]
        if mst_list:
            best_mst = max(
                mst_list,
                key=lambda t: t.get("weight", t.get("similarity_score", 0)),
            )
            most_similar = best_mst.get("task_name")

    if not most_similar and "task_similarity" in parsed_json:
        ts_list = parsed_json["task_similarity"]
        best_ts = max(ts_list, key=lambda t: t.get("overall_similarity", 0))
        most_similar = best_ts.get("known_task")

    if not most_similar:
        most_similar = "locomotion"  # safe fallback

    return best_cluster, most_similar


# ---------------------------------------------------------------------------
# Helper: save recommendation in standard format
# ---------------------------------------------------------------------------

def save_recommendation(parsed_json, response_text, new_task, model,
                        temperature, usage, system_prompt, user_prompt):
    """Save LLM recommendation JSON in the standard format.

    Uses new_task (not 'custom') in the filename so load_recommendation()
    can find it later with --skip_llm.
    """
    os.makedirs(_REC_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        _REC_OUTPUT_DIR, f"recommendation_{new_task}_{timestamp}.json"
    )

    result = {
        "metadata": {
            "timestamp": timestamp,
            "model": model,
            "new_task": new_task,
            "temperature": temperature,
            "usage": usage,
            "source": "end_to_end_demo.py",
        },
        "prompts": {
            "system": system_prompt,
            "user": user_prompt,
        },
        "raw_response": response_text,
        "parsed_recommendation": parsed_json,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path, timestamp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end LLM-guided morphology design demo"
    )
    parser.add_argument("--new_task", type=str, default="manipulation_ball",
                        help="New unseen task (default: manipulation_ball)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM call, reuse saved recommendation")
    parser.add_argument("--recommendation", type=str, default=None,
                        help="Path to specific recommendation JSON to load")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Override recommended cluster ID")
    parser.add_argument("--similar_task", type=str, default=None,
                        choices=list(TASK_DIRS.keys()),
                        help="Override most similar task")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Claude model for LLM step")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Interpolation weight (0=elite1, 1=elite2)")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Alternative training data root")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output at each step")
    args = parser.parse_args()

    # Setup output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    xml_dir = os.path.join(args.output_dir, "xmls")
    os.makedirs(xml_dir, exist_ok=True)
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure MUJOCO_GL is set for headless rendering
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    print("=" * 70)
    print("  End-to-End LLM-Guided Morphology Design Demo")
    print("=" * 70)
    print(f"  New task: {args.new_task}")
    print(f"  Skip LLM: {args.skip_llm}")

    # ==================================================================
    # STEP 1: LLM Similarity Reasoning
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 1: LLM Task Similarity Reasoning")
    print(f"{'='*70}")

    parsed_json = None
    rec_path = None
    best_cluster = args.cluster
    most_similar = args.similar_task

    # If both cluster and task are manually specified, skip LLM entirely
    if best_cluster is not None and most_similar is not None:
        print(f"  Using manual overrides: cluster={best_cluster}, "
              f"task={most_similar}")
        print("  (Skipping LLM step)")
    elif args.skip_llm:
        # --- Load existing recommendation ---
        if args.recommendation:
            print(f"  Loading recommendation from: {args.recommendation}")
            with open(args.recommendation) as f:
                rec_data = json.load(f)
            parsed_json = rec_data.get("parsed_recommendation", {})
            rec_path = args.recommendation
        else:
            print(f"  Loading most recent recommendation for '{args.new_task}'...")
            try:
                best_cluster, most_similar, rec_data = load_recommendation(
                    args.new_task, _PROJECT_ROOT
                )
                parsed_json = rec_data.get("parsed_recommendation", {})
                rec_path = "(loaded via load_recommendation)"
                print(f"  Found: cluster={best_cluster}, task={most_similar}")
            except FileNotFoundError as e:
                print(f"\n  ERROR: {e}")
                print("  Run without --skip_llm first to generate a recommendation.")
                sys.exit(1)
    else:
        # --- Call Claude API ---
        # Check API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n  ERROR: ANTHROPIC_API_KEY not set.")
            print("  Either: export ANTHROPIC_API_KEY='your-key'")
            print("  Or run with --skip_llm to use a saved recommendation.")
            sys.exit(1)

        # Get task description
        if args.new_task in ALL_TASK_DESCRIPTIONS:
            task_description = ALL_TASK_DESCRIPTIONS[args.new_task]
        else:
            print(f"\n  ERROR: No description found for task '{args.new_task}'")
            print(f"  Available: {list(ALL_TASK_DESCRIPTIONS.keys())}")
            sys.exit(1)

        # Load performance data
        print(f"  Loading performance data from {_BEST_PERFORMERS_PATH}")
        with open(_BEST_PERFORMERS_PATH) as f:
            perf_data = json.load(f)
        print(f"  Loaded {len(perf_data)} entries")

        # Build prompt using "custom" path with our task description
        user_prompt = build_user_prompt(
            perf_data, "custom", custom_description=task_description
        )

        if args.verbose:
            print(f"\n  User prompt length: {len(user_prompt)} chars")
            print(f"  First 500 chars:\n{user_prompt[:500]}")

        # Call Claude API
        print(f"\n  Calling {args.model} (temperature={args.temperature})...")
        try:
            message = call_claude(
                SYSTEM_PROMPT,
                user_prompt,
                model=args.model,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"\n  ERROR: API call failed: {e}")
            print("  Try running with --skip_llm if you have a saved recommendation.")
            sys.exit(1)

        response_text = message.content[0].text
        usage = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }
        print(f"  Response: {usage['input_tokens']} input, "
              f"{usage['output_tokens']} output tokens")

        if args.verbose:
            print(f"\n  Full response:\n{response_text[:2000]}...")

        # Parse JSON
        parsed_json = extract_json_from_response(response_text)
        if parsed_json is None:
            print("\n  ERROR: Could not parse JSON from LLM response.")
            # Save raw response for debugging
            debug_path = os.path.join(
                args.output_dir, f"failed_response_{timestamp}.txt"
            )
            with open(debug_path, "w") as f:
                f.write(response_text)
            print(f"  Raw response saved to: {debug_path}")
            sys.exit(1)

        # Save recommendation with correct task name in filename
        rec_path, _ = save_recommendation(
            parsed_json, response_text, args.new_task, args.model,
            args.temperature, usage, SYSTEM_PROMPT, user_prompt,
        )
        print(f"  Recommendation saved: {rec_path}")

    # --- Extract cluster and task from recommendation (if not already set) ---
    if parsed_json and (best_cluster is None or most_similar is None):
        rec_cluster, rec_task = extract_recommendation_info(parsed_json)
        if best_cluster is None:
            best_cluster = rec_cluster
        if most_similar is None:
            most_similar = rec_task

    # Print LLM reasoning summary
    print(f"\n  Recommended: Cluster {best_cluster} x {most_similar}")
    if parsed_json and "task_similarity" in parsed_json:
        print("  Task Similarity Scores:")
        for ts in parsed_json["task_similarity"]:
            task = ts.get("known_task", "?")
            score = ts.get("overall_similarity", "?")
            print(f"    {task}: {score}")
    if parsed_json and "most_similar_tasks" in parsed_json:
        print("  Multi-Task Similarity:")
        for mst in parsed_json["most_similar_tasks"]:
            name = mst.get("task_name", "?")
            score = mst.get("similarity_score", "?")
            weight = mst.get("weight", "?")
            print(f"    {name}: score={score}, weight={weight}")

    # ==================================================================
    # STEP 2: Load VAE Model
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 2: Load VAE Model")
    print(f"{'='*70}")

    model = load_vae_model(VAE_CHECKPOINT, device)
    print("  VAE loaded")

    # ==================================================================
    # STEP 3: Load Elite Pool
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 3: Load Elite Pool")
    print(f"{'='*70}")
    print(f"  Source: Cluster {best_cluster} x {most_similar}")

    pool = load_task_elite_pool(
        best_cluster, most_similar, model, device, data_root=args.data_root
    )
    print(f"  Total elites loaded: {len(pool)}")
    if pool:
        rewards = [a["reward"] for a in pool]
        print(f"  Reward range: [{min(rewards):.1f}, {max(rewards):.1f}]")

    # ==================================================================
    # STEP 4: Deduplicate & Select Top-2
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 4: Deduplicate & Select Top-2 Unique Elites")
    print(f"{'='*70}")

    unique_pool = deduplicate_pool(pool)
    print(f"  Unique elites after MD5 dedup: {len(unique_pool)}")

    if len(unique_pool) < 2:
        print("\n  ERROR: Need at least 2 unique elites for interpolation.")
        print(f"  Only found {len(unique_pool)} unique morphologies.")
        sys.exit(1)

    top2 = unique_pool[:2]
    seq1 = get_sequence_length(top2[0]["vec"].numpy())
    seq2 = get_sequence_length(top2[1]["vec"].numpy())
    print(f"\n  Top-2 unique elites:")
    print(f"    #1: Agent {top2[0]['agent_id']}, "
          f"reward={top2[0]['reward']:.1f}, {seq1} limbs")
    print(f"    #2: Agent {top2[1]['agent_id']}, "
          f"reward={top2[1]['reward']:.1f}, {seq2} limbs")

    # ==================================================================
    # STEP 5: VAE Latent Midpoint Interpolation
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 5: VAE Latent Space Midpoint Interpolation")
    print(f"{'='*70}")

    z1 = top2[0]["z"]  # [1, 11, 32]
    z2 = top2[1]["z"]  # [1, 11, 32]
    z_mid = (1 - args.alpha) * z1 + args.alpha * z2
    l2_dist = torch.norm(z1 - z2).item()
    print(f"  L2 distance(z1, z2): {l2_dist:.4f}")

    decoded_vec = decode_from_latent(model, z_mid, device)
    xml_latent = vec_to_xml_safe(decoded_vec[0])
    latent_valid = xml_latent is not None
    print(f"  Latent midpoint: {'VALID' if latent_valid else 'INVALID'}")

    latent_xml_path = None
    if latent_valid:
        latent_xml_path = os.path.join(
            xml_dir, f"latent_midpoint_{args.new_task}.xml"
        )
        with open(latent_xml_path, "w") as f:
            f.write(xml_latent)
        print(f"  Saved: {latent_xml_path}")

    # ==================================================================
    # STEP 6: Raw-Space Midpoint Interpolation
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 6: Raw Design Space Midpoint Interpolation")
    print(f"{'='*70}")

    vec1 = top2[0]["vec"].numpy().astype(np.float64)
    vec2 = top2[1]["vec"].numpy().astype(np.float64)

    # Build masks (active rows = True, padded rows = False)
    mask1 = np.ones_like(vec1, dtype=bool)
    mask1[seq1:, :] = False
    mask2 = np.ones_like(vec2, dtype=bool)
    mask2[seq2:, :] = False

    result_len = min(seq1, seq2)
    print(f"  Parent 1: {seq1} limbs, Parent 2: {seq2} limbs")
    print(f"  Result: {result_len} limbs (round down)")

    vec_raw, mask_raw = interpolate_raw(
        vec1, vec2, mask1, mask2,
        alpha=args.alpha,
        reward1=top2[0]["reward"],
        reward2=top2[1]["reward"],
    )

    vec_raw_tensor = torch.tensor(vec_raw, dtype=torch.float32)
    xml_raw = vec_to_xml_safe(vec_raw_tensor)
    raw_valid = xml_raw is not None
    print(f"  Raw midpoint: {'VALID' if raw_valid else 'INVALID'}")

    raw_xml_path = None
    if raw_valid:
        raw_xml_path = os.path.join(
            xml_dir, f"raw_midpoint_{args.new_task}.xml"
        )
        with open(raw_xml_path, "w") as f:
            f.write(xml_raw)
        print(f"  Saved: {raw_xml_path}")

    # ==================================================================
    # STEP 7: Render All 4 Morphologies
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 7: Rendering Morphologies")
    print(f"{'='*70}")

    from PIL import Image

    W, H = args.width, args.height

    # --- Elite 1 ---
    try:
        px1 = render_morphology_from_xml_path(top2[0]["xml_path"], W, H)
        r_str = f"R={top2[0]['reward']:.0f}" if top2[0]["reward"] > 0 else "R=?"
        px1 = add_text_to_image(px1, [
            f"Elite #1 ({top2[0]['agent_id']})",
            f"{r_str}, {seq1} limbs",
        ])
        print("  Elite #1: rendered")
    except Exception as e:
        print(f"  Elite #1: render failed ({e})")
        px1 = create_invalid_placeholder(W, H)

    # --- Elite 2 ---
    try:
        px2 = render_morphology_from_xml_path(top2[1]["xml_path"], W, H)
        r_str = f"R={top2[1]['reward']:.0f}" if top2[1]["reward"] > 0 else "R=?"
        px2 = add_text_to_image(px2, [
            f"Elite #2 ({top2[1]['agent_id']})",
            f"{r_str}, {seq2} limbs",
        ])
        print("  Elite #2: rendered")
    except Exception as e:
        print(f"  Elite #2: render failed ({e})")
        px2 = create_invalid_placeholder(W, H)

    # --- Latent Midpoint ---
    if latent_valid:
        try:
            px_lat = render_from_xml_string(xml_latent, W, H)
            latent_seq = get_sequence_length(
                decoded_vec[0].detach().cpu().numpy()
            )
            px_lat = add_text_to_image(px_lat, [
                "Latent Midpoint",
                f"L2={l2_dist:.2f}, {latent_seq} limbs",
            ])
            print("  Latent midpoint: rendered")
        except Exception as e:
            print(f"  Latent midpoint: render failed ({e})")
            px_lat = create_invalid_placeholder(W, H)
            latent_valid = False
    else:
        px_lat = create_invalid_placeholder(W, H)

    # --- Raw Midpoint ---
    if raw_valid:
        try:
            px_raw = render_from_xml_string(xml_raw, W, H)
            px_raw = add_text_to_image(px_raw, [
                "Raw Midpoint",
                f"{result_len} limbs (round-down)",
            ])
            print("  Raw midpoint: rendered")
        except Exception as e:
            print(f"  Raw midpoint: render failed ({e})")
            px_raw = create_invalid_placeholder(W, H)
            raw_valid = False
    else:
        px_raw = create_invalid_placeholder(W, H)

    # ==================================================================
    # STEP 8: Composite Visualization
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 8: Building Composite Visualization")
    print(f"{'='*70}")

    # Row 1: Top-2 elites
    strip_elites = create_strip_image(
        [px1, px2],
        [f"Elite #1 ({seq1} limbs)", f"Elite #2 ({seq2} limbs)"],
        strip_label=(
            f"C{best_cluster} x {most_similar} -> {args.new_task}: "
            f"Top-2 Unique Elites"
        ),
        cell_width=W,
        cell_height=H,
    )

    # Row 2: Interpolated morphologies
    strip_midpoints = create_strip_image(
        [px_lat, px_raw],
        [
            f"Latent Midpoint (L2={l2_dist:.2f})",
            f"Raw Midpoint ({result_len} limbs)",
        ],
        strip_label=(
            f"Interpolated Morphologies (alpha={args.alpha})"
        ),
        cell_width=W,
        cell_height=H,
    )

    # Stack vertically
    composite = np.concatenate([strip_elites, strip_midpoints], axis=0)

    out_path = os.path.join(
        args.output_dir,
        f"end_to_end_{args.new_task}_{timestamp}.png",
    )
    Image.fromarray(composite).save(out_path)
    print(f"  Visualization saved: {out_path}")

    # ==================================================================
    # STEP 9: Save Metadata JSON
    # ==================================================================
    print(f"\n{'='*70}")
    print("  Step 9: Saving Metadata")
    print(f"{'='*70}")

    meta = {
        "metadata": {
            "timestamp": timestamp,
            "new_task": args.new_task,
            "cli_args": {
                "skip_llm": args.skip_llm,
                "cluster_override": args.cluster,
                "similar_task_override": args.similar_task,
                "model": args.model,
                "temperature": args.temperature,
                "alpha": args.alpha,
            },
        },
        "llm_recommendation": {
            "best_cluster": best_cluster,
            "most_similar_task": most_similar,
            "recommendation_file": rec_path,
        },
        "elite_pool": {
            "total_loaded": len(pool),
            "unique_after_dedup": len(unique_pool),
            "reward_range": [
                min(a["reward"] for a in pool),
                max(a["reward"] for a in pool),
            ] if pool else [],
        },
        "top_2_elites": [
            {
                "agent_id": top2[0]["agent_id"],
                "reward": top2[0]["reward"],
                "sequence_length": seq1,
                "xml_path": top2[0]["xml_path"],
            },
            {
                "agent_id": top2[1]["agent_id"],
                "reward": top2[1]["reward"],
                "sequence_length": seq2,
                "xml_path": top2[1]["xml_path"],
            },
        ],
        "latent_midpoint": {
            "valid": latent_valid,
            "l2_distance": l2_dist,
            "xml_path": latent_xml_path,
        },
        "raw_midpoint": {
            "valid": raw_valid,
            "sequence_length": result_len,
            "strategy": "round_down_limbs_AND_joints",
            "xml_path": raw_xml_path,
        },
        "visualization_path": out_path,
        "full_recommendation": parsed_json,
    }

    # Add similarity scores to metadata if available
    if parsed_json and "task_similarity" in parsed_json:
        meta["llm_recommendation"]["similarity_scores"] = {
            ts["known_task"]: ts["overall_similarity"]
            for ts in parsed_json["task_similarity"]
        }

    meta_path = os.path.join(
        args.output_dir,
        f"end_to_end_{args.new_task}_{timestamp}.json",
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata saved: {meta_path}")

    # ==================================================================
    # STEP 10: Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("  DEMO COMPLETE")
    print(f"{'='*70}")
    print(f"  Task:              {args.new_task}")
    print(f"  Recommended:       Cluster {best_cluster} x {most_similar}")
    print(f"  Elite pool:        {len(pool)} total, "
          f"{len(unique_pool)} unique")
    print(f"  Top-2 rewards:     {top2[0]['reward']:.1f}, "
          f"{top2[1]['reward']:.1f}")
    print(f"  Latent midpoint:   {'VALID' if latent_valid else 'INVALID'}")
    print(f"  Raw midpoint:      {'VALID' if raw_valid else 'INVALID'}")
    print(f"  Visualization:     {out_path}")
    print(f"  Metadata:          {meta_path}")
    if rec_path and not args.skip_llm:
        print(f"  Recommendation:    {rec_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
