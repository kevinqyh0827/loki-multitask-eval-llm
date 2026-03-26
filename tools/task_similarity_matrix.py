#!/usr/bin/env python3
"""Build an N x N task similarity matrix using LLM analysis.

Sends all task descriptions to Claude API in a single prompt,
asks for pairwise similarity scores along 6 physical dimensions,
and outputs a symmetric matrix as JSON, CSV, and heatmap PNG.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python tools/task_similarity_matrix.py
    python tools/task_similarity_matrix.py --model claude-opus-4-6
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "task_similarity")

sys.path.insert(0, _PROJECT_ROOT)

# Reuse the system prompt from the recommender
from tools.llm_cluster_recommender import SYSTEM_PROMPT, call_claude, extract_json_from_response

# ---------------------------------------------------------------------------
# ALL task descriptions (9 tasks)
# ---------------------------------------------------------------------------
ALL_TASK_DESCRIPTIONS = {
    "locomotion": """### Task: Locomotion (Flat Terrain)
- **Environment**: Large flat floor (100x100 units), no obstacles or terrain variation. Modules: Agent + Floor.
- **Goal**: Move forward (positive x-direction) as fast as possible.
- **Reward**: `r = 1.0 * x_velocity`. No control cost penalty (weight = 0.0). No standing bonus (weight = 0.0).
- **Termination**: Episode ends if torso height drops below 50% of initial standing height.
- **Observation**: Proprioceptive only (joint angles, joint velocities, torso orientation). No terrain/height-field sensing.
- **Key morphological demands**: Pure speed task on flat ground. Favors efficient gaits, long stride, dynamic balance, low energy waste. Limb arrangement for smooth, fast forward locomotion is critical.""",

    "obstacle": """### Task: Obstacle Course (50 obstacles)
- **Environment**: Flat terrain (50x20 units) with 50 randomly placed obstacles of varying dimensions. Boundary walls on both sides. 3-unit obstacle-free starting zone. Modules: Agent + Terrain + Objects.
- **Goal**: Move forward (positive x-direction) as fast as possible while navigating around or over obstacles.
- **Reward**: `r = 1.0 * x_velocity`. Pure forward speed, no control cost or standing bonus.
- **Termination**: Episode ends if torso height drops below 50% of initial height, or if agent contacts terrain edge.
- **Observation**: Proprioceptive + 2D height-field observation (sensing grid: 1 unit behind, 4 ahead, 4 left, 4 right).
- **Key morphological demands**: Speed AND terrain navigation. Agile enough to navigate around obstacles while maintaining forward progress. Benefits from compact body, flexible joints, good sensorimotor integration for height-field observations.""",

    "many_obstacle": """### Task: Dense Obstacle Course (150 obstacles)
- **Environment**: Flat terrain (50x20 units) with 150 randomly placed obstacles — 3x denser than the standard obstacle course. Boundary walls on both sides. 3-unit obstacle-free starting zone. Modules: Agent + Terrain + Objects.
- **Goal**: Move forward (positive x-direction) as fast as possible through very dense obstacle field.
- **Reward**: `r = 1.0 * x_velocity`. Pure forward speed, same as obstacle task.
- **Termination**: Episode ends if torso height drops below 50% of initial height, or if agent contacts terrain edge.
- **Observation**: Proprioceptive + 2D height-field observation (sensing grid: 1 unit behind, 4 ahead, 4 left, 4 right).
- **Key morphological demands**: Similar to obstacle but much denser obstacles mean less room to maneuver. Favors more compact morphologies that can fit through narrow gaps. High obstacle density means the agent must constantly adapt its gait. Requires robust collision handling and the ability to maintain forward progress despite frequent obstacle contacts.""",

    "bump": """### Task: Bumpy Terrain Traversal
- **Environment**: Flat base terrain (50x20 units) densely covered with 250 small bumps. Boundary walls on both sides. Modules: Agent + Terrain + Objects.
- **Bumps**: Each bump has length 0.8-1.6 units, width 0.8-1.6 units, height 0.1-0.25 units. Densely scattered with a 5-unit buffer at start. Many small terrain irregularities rather than large obstacles.
- **Goal**: Move forward (positive x-direction) as fast as possible over the bumpy terrain.
- **Reward**: `r = 1.0 * x_velocity`. Pure forward speed, no control cost or standing bonus.
- **Termination**: Episode ends if torso height drops below 50% of initial height, or if agent contacts terrain edge.
- **Observation**: Proprioceptive + 2D height-field observation (sensing grid: 1 unit behind, 4 ahead, 4 left, 4 right).
- **Key morphological demands**: Hybrid between locomotion (forward speed) and obstacle (terrain awareness). Bumps are small but extremely dense — agent cannot avoid them, must traverse THROUGH them. Favors good shock absorption, stable foothold on uneven surfaces, adaptive gait for continuous small perturbations, robust balance under persistent terrain disturbances.""",

    "incline": """### Task: Incline (Slope Traversal)
- **Environment**: Curved slope terrain (75x20 units) at -10 degree angle (downhill grade). Smooth sinusoidal curve profile. No boundary walls. Modules: Agent + Terrain.
- **Goal**: Move forward (positive x-direction) along the slope as fast as possible while maintaining balance.
- **Reward**: `r = (1.0 * x_velocity) / cos(10deg) + 0.2 * min(torso_height, 2 * initial_height)`. Forward reward normalized by cos(angle) for slope difficulty. Standing bonus (weight = 0.2) rewards maintaining torso height.
- **Termination**: Episode ends if torso height drops below 20% of initial height (stricter than 50% for flat tasks), or if agent goes past terrain edge.
- **Observation**: Proprioceptive only (joint angles, velocities, torso orientation). No height-field sensing.
- **Key morphological demands**: Most physically demanding task. -10 degree slope creates gravitational forces pulling agent sideways/downhill. Requires strong limbs for climbing/braking, low center of gravity for slope stability, body proportions that resist toppling on inclines, ability to maintain upright posture. Favors robust, powerful builds over lightweight speed-optimized ones.""",

    "push_box_incline": """### Task: Push Box on Incline
- **Environment**: Curved slope terrain (40x20 units) at -10 degree angle, with a 2kg movable box at position (-35, 0) and goal zone at (35, 0). No boundary walls. Modules: Agent + Terrain + Objects.
- **Goal**: Two-phase manipulation: (1) Navigate to box and make contact, (2) push box from start to goal position, all on a -10 degree slope.
- **Reward**: `r = reach_reward + push_reward - ctrl_cost` where reach_reward = (distance_reduction) * 100 + 10 bonus on contact; push_reward = (goal_distance_reduction) * 100 + 10 bonus on reaching goal.
- **Termination**: Episode ends if torso height drops below 20% of initial height.
- **Observation**: Proprioceptive only. No height-field sensing.
- **Key morphological demands**: Fundamentally different from locomotion tasks — requires MANIPULATION on a slope. Agent must navigate to specific positions, exert pushing force on 2kg object, maintain stability while applying force on slope, and control body precisely for object interaction. Favors strong, stable builds, ability to generate directed force, body shape suitable for pushing, and precise locomotion control.""",

    "exploration": """### Task: Exploration (Area Coverage)
- **Environment**: Variable terrain with boundary walls. Modules: Agent + Terrain + Objects.
- **Goal**: Explore terrain by moving in any direction as fast as possible — not just forward.
- **Reward**: `r = FORWARD_REWARD_WEIGHT * ||xy_velocity||_2 - ctrl_cost`. Uses 2D velocity NORM (Euclidean distance of x,y velocity), not just x-velocity. Rewards movement in ALL directions equally.
- **Termination**: Episode ends if torso height drops below 50% of initial height.
- **Observation**: Proprioceptive + height-field via ExploreTerrainReward wrapper.
- **Key morphological demands**: Unlike locomotion which rewards only forward speed, this rewards multi-directional movement. Requires morphologies that can move efficiently in ANY direction, not just forward. Favors balanced, versatile builds with good turning ability, symmetric body plans, and adaptability to varied terrain. Morphologies optimized for forward-only locomotion may underperform.""",

    "patrol": """### Task: Patrol (Goal Reaching)
- **Environment**: Flat floor with two toggling patrol goal positions (PATROL_HALF_LEN apart). Modules: Agent + Floor + PatrolGoals.
- **Goal**: Navigate to goal position, then toggle to next goal and repeat. Metric = number of successful goal toggles.
- **Reward**: `r = -ctrl_cost + reach_reward` where reach_reward (from ReachReward wrapper) provides distance-based shaping. NO forward velocity reward at all — purely about reaching specific positions.
- **Termination**: Episode ends only if torso height drops below threshold.
- **Observation**: Proprioceptive only. No height-field sensing.
- **Key morphological demands**: Navigation/reaching task, fundamentally different from speed-based locomotion. No forward speed bonus — success measured by goal toggles, not distance traveled. Requires precision locomotion control, good maneuverability, ability to stop and orient toward goals. Favors morphologies with responsive control, good turning radius, and ability to decelerate smoothly. Directional control is more important than raw speed.""",

    "manipulation_ball": """### Task: Ball Manipulation (Flat Terrain)
- **Environment**: Flat floor (15x15 units) with a ball (radius 0.20) and a goal position. Modules: Agent + Floor + Objects.
- **Goal**: Two-phase manipulation: (1) Navigate to ball and make contact (using multi-site distance from agent limbs), (2) push ball to goal position.
- **Reward**: `r = reach_reward + push_reward - ctrl_cost` where reach_reward = (agent-ball distance reduction) * 100 + 10 bonus on contact; push_reward = (ball-goal distance reduction) * 100 + 10 bonus on reaching goal.
- **Termination**: Episode ends if torso height drops below 50% of initial height.
- **Observation**: Proprioceptive only. No height-field sensing.
- **Key morphological demands**: Multi-stage manipulation task on FLAT terrain (unlike push_box_incline which is on a slope). Requires accurate distance estimation, force application in specific direction, precision positioning relative to object. Ball is harder to push than box (rolls). Favors morphologies with multiple limbs for object manipulation, strong body for force transmission, and good proprioceptive feedback for contact sensing. Less emphasis on slope stability compared to push_box_incline.""",
}

TASK_ORDER = [
    "locomotion", "obstacle", "many_obstacle", "bump",
    "incline", "push_box_incline",
    "exploration", "patrol", "manipulation_ball",
]

SIMILARITY_DIMENSIONS = [
    "terrain", "reward_structure", "balance_requirements",
    "speed_vs_robustness", "sensory_demands", "contact_pattern",
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_similarity_prompt(tasks):
    """Build prompt asking for pairwise similarity analysis."""
    sections = []

    sections.append("""## Task Similarity Analysis Request

You are given descriptions of N robot locomotion/manipulation tasks in a MuJoCo physics simulation. \
Your job is to compute a pairwise similarity matrix between ALL task pairs.

## Task Descriptions
""")

    for i, task in enumerate(tasks):
        sections.append(f"**Task {i+1}: {task}**\n")
        sections.append(ALL_TASK_DESCRIPTIONS[task])
        sections.append("")

    # Build pair list
    pairs = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            pairs.append((tasks[i], tasks[j]))

    sections.append(f"""
## Instructions

For each of the {len(pairs)} unique task pairs, provide:
1. An overall similarity score (0.0 = completely different, 1.0 = identical)
2. Six dimension-specific scores (each 0.0 to 1.0):
   - **terrain**: How similar are the terrain types and complexity?
   - **reward_structure**: How similar are the reward functions and objectives?
   - **balance_requirements**: How similar are the stability/balance demands?
   - **speed_vs_robustness**: Do both tasks favor speed or robustness equally?
   - **sensory_demands**: How similar are the observation requirements?
   - **contact_pattern**: How similar are the ground contact and gait patterns needed?

Compute the overall similarity as a weighted combination of the dimensions, \
with physical reasoning about which dimensions matter most for each pair.

**IMPORTANT**: Output your response as a JSON object with exactly this schema:
```json
{{
  "task_pairs": [
    {{
      "task_a": "<task name>",
      "task_b": "<task name>",
      "overall_similarity": <float 0.0-1.0>,
      "dimensions": {{
        "terrain": <float>,
        "reward_structure": <float>,
        "balance_requirements": <float>,
        "speed_vs_robustness": <float>,
        "sensory_demands": <float>,
        "contact_pattern": <float>
      }},
      "reasoning": "<1-2 sentences explaining the similarity score>"
    }}
  ]
}}
```

Include ALL {len(pairs)} pairs. The task pairs are:
""")

    for i, (a, b) in enumerate(pairs):
        sections.append(f"  {i+1}. {a} vs {b}")

    sections.append("""
Wrap the JSON in a markdown code block (```json ... ```). You may include brief reasoning before the JSON.""")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Parsing and matrix construction
# ---------------------------------------------------------------------------

def build_matrix(parsed_json, tasks):
    """Convert parsed pairs into NxN similarity matrices."""
    n = len(tasks)
    task_idx = {t: i for i, t in enumerate(tasks)}

    # Overall matrix
    overall = np.eye(n)

    # Per-dimension matrices
    dim_matrices = {d: np.eye(n) for d in SIMILARITY_DIMENSIONS}

    pairs = parsed_json.get("task_pairs", [])
    for pair in pairs:
        a = pair.get("task_a", "")
        b = pair.get("task_b", "")
        if a not in task_idx or b not in task_idx:
            continue
        i, j = task_idx[a], task_idx[b]
        sim = float(pair.get("overall_similarity", 0))
        overall[i, j] = sim
        overall[j, i] = sim

        dims = pair.get("dimensions", {})
        for d in SIMILARITY_DIMENSIONS:
            val = float(dims.get(d, 0))
            dim_matrices[d][i, j] = val
            dim_matrices[d][j, i] = val

    return overall, dim_matrices


def plot_heatmap(matrix, tasks, title, output_path):
    """Plot a single heatmap."""
    n = len(tasks)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tasks, fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > 0.65 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build N x N task similarity matrix")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Claude model to use")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = TASK_ORDER
    n = len(tasks)
    n_pairs = n * (n - 1) // 2

    print(f"Building {n}x{n} task similarity matrix ({n_pairs} unique pairs)")
    print(f"Tasks: {tasks}")

    # Build prompt
    user_prompt = build_similarity_prompt(tasks)
    system_prompt = SYSTEM_PROMPT

    # Call API
    print(f"\nCalling {args.model} (max_tokens={args.max_tokens}, temperature={args.temperature})...")
    message = call_claude(system_prompt, user_prompt,
                          model=args.model, max_tokens=args.max_tokens,
                          temperature=args.temperature)

    response_text = message.content[0].text
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
    print(f"  Response: {usage['input_tokens']} input, {usage['output_tokens']} output tokens")

    # Parse JSON
    parsed_json = extract_json_from_response(response_text)
    if parsed_json is None:
        print("\nERROR: Could not parse JSON from response.")
        # Save raw response for debugging
        raw_path = os.path.join(args.output_dir, "raw_response.txt")
        with open(raw_path, "w") as f:
            f.write(response_text)
        print(f"  Raw response saved to: {raw_path}")
        return

    pairs = parsed_json.get("task_pairs", [])
    print(f"  Parsed {len(pairs)}/{n_pairs} task pairs")

    # Build matrices
    overall_matrix, dim_matrices = build_matrix(parsed_json, tasks)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Full JSON result
    result = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "num_tasks": n,
            "task_order": tasks,
            "dimensions": SIMILARITY_DIMENSIONS,
            "usage": usage,
        },
        "raw_response": response_text,
        "parsed_pairs": parsed_json,
        "overall_matrix": overall_matrix.tolist(),
        "dimension_matrices": {d: m.tolist() for d, m in dim_matrices.items()},
    }
    json_path = os.path.join(args.output_dir, f"similarity_matrix_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON saved: {json_path}")

    # 2. CSV matrix
    csv_path = os.path.join(args.output_dir, f"similarity_matrix_{timestamp}.csv")
    with open(csv_path, "w") as f:
        f.write("," + ",".join(tasks) + "\n")
        for i, task in enumerate(tasks):
            row = [f"{overall_matrix[i, j]:.3f}" for j in range(n)]
            f.write(f"{task}," + ",".join(row) + "\n")
    print(f"  CSV saved: {csv_path}")

    # 3. Heatmap — overall
    heatmap_path = os.path.join(args.output_dir, f"similarity_heatmap_{timestamp}.png")
    plot_heatmap(overall_matrix, tasks,
                 "Task Similarity Matrix (Overall)", heatmap_path)
    print(f"  Heatmap saved: {heatmap_path}")

    # 4. Per-dimension heatmaps
    for dim in SIMILARITY_DIMENSIONS:
        dim_path = os.path.join(args.output_dir, f"similarity_{dim}_{timestamp}.png")
        plot_heatmap(dim_matrices[dim], tasks,
                     f"Task Similarity: {dim.replace('_', ' ').title()}", dim_path)
    print(f"  Dimension heatmaps saved ({len(SIMILARITY_DIMENSIONS)} files)")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TASK SIMILARITY MATRIX (Overall)")
    print("=" * 70)

    # Print matrix
    header = f"{'':>20} |"
    for t in tasks:
        header += f" {t[:10]:>10} |"
    print(header)
    print("-" * len(header))
    for i, task in enumerate(tasks):
        row = f"{task:>20} |"
        for j in range(n):
            row += f" {overall_matrix[i, j]:>10.2f} |"
        print(row)

    # Most/least similar pairs
    print("\nTop 5 most similar pairs:")
    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_scores.append((tasks[i], tasks[j], overall_matrix[i, j]))
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    for a, b, s in pair_scores[:5]:
        print(f"  {a} <-> {b}: {s:.3f}")

    print("\nTop 5 least similar pairs:")
    for a, b, s in pair_scores[-5:]:
        print(f"  {a} <-> {b}: {s:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
