#!/usr/bin/env python3
"""
LLM-guided morphology cluster recommendation for unseen tasks.

Loads the cluster-task performance table from completed LOKI training runs,
composes a structured prompt describing known tasks and their performance,
then asks Claude to predict which clusters would be best for a new unseen task.

Usage:
    export ANTHROPIC_API_KEY="your-key"

    # Predict best cluster for the bump task
    python tools/llm_cluster_recommender.py --new_task bump

    # Predict best cluster for push_box_incline
    python tools/llm_cluster_recommender.py --new_task push_box_incline

    # Custom task description
    python tools/llm_cluster_recommender.py --new_task custom \
        --custom_description "Navigate narrow corridors requiring compact body shapes"

    # Use a different model
    python tools/llm_cluster_recommender.py --new_task bump --model claude-opus-4-6
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import anthropic

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
_BEST_PERFORMERS_PATH = os.path.join(
    _RESULTS_DIR, "best_performers", "best_performers_summary.json"
)
_DEFAULT_OUTPUT_DIR = os.path.join(_RESULTS_DIR, "llm_recommendations")

# ---------------------------------------------------------------------------
# Known task descriptions (derived from actual env source code)
# ---------------------------------------------------------------------------
KNOWN_TASK_DESCRIPTIONS = {
    "locomotion": """### Task: Locomotion (Flat Terrain)
- **Environment**: Large flat floor (100x100 units), no obstacles or terrain variation. Modules: Agent + Floor.
- **Goal**: Move forward (positive x-direction) as fast as possible.
- **Reward**: `r = 1.0 * x_velocity`. No control cost penalty (weight = 0.0). No standing bonus (weight = 0.0).
- **Termination**: Episode ends if torso height drops below 50% of initial standing height.
- **Observation**: Proprioceptive only (joint angles, joint velocities, torso orientation). No terrain/height-field sensing.
- **Key morphological demands**: This is a pure speed task on flat ground. Favors morphologies with efficient gaits, long effective stride length, good dynamic balance, and low energy waste. Limb arrangement that enables smooth, fast forward locomotion is critical. Body mass should be well-distributed for stable high-speed movement.""",

    "obstacle": """### Task: Obstacle Course
- **Environment**: Flat terrain (50x20 units) with randomly placed obstacles. Boundary walls on both sides prevent lateral escape. Modules: Agent + Terrain + Objects.
- **Obstacles**: Multiple obstacles of varying dimensions (length and width vary, height varies) scattered across the terrain. The first 3 units are flat (obstacle-free starting zone).
- **Goal**: Move forward (positive x-direction) as fast as possible while navigating around or over obstacles.
- **Reward**: `r = 1.0 * x_velocity`. Same reward structure as locomotion — pure forward speed. No control cost or standing bonus.
- **Termination**: Episode ends if torso height drops below 50% of initial height, or if agent contacts terrain edge.
- **Observation**: Proprioceptive + 2D height-field observation (sensing grid: 1 unit behind, 4 ahead, 4 left, 4 right). The agent can "see" upcoming terrain features.
- **Key morphological demands**: Requires both speed AND terrain navigation ability. Morphologies must be agile enough to navigate around obstacles while maintaining forward progress. Benefits from: compact body that fits through gaps, flexible joints for adaptive stepping, good sensorimotor integration to use height-field observations, and stability when contacting obstacles.""",

    "incline": """### Task: Incline (Slope Traversal)
- **Environment**: Curved slope terrain (75x20 units) at -10 degree angle (downhill grade). The terrain follows a smooth sinusoidal curve profile. No boundary walls. Modules: Agent + Terrain.
- **Goal**: Move forward (positive x-direction) along the slope as fast as possible while maintaining balance.
- **Reward**: `r = (1.0 * x_velocity) / cos(10deg) + 0.2 * min(torso_height, 2 * initial_height)`. The forward reward is normalized by cos(angle) to account for slope difficulty. A standing bonus (weight = 0.2) rewards maintaining torso height, emphasizing balance.
- **Termination**: Episode ends if torso height drops below 20% of initial height (stricter than locomotion's 50%), or if agent goes past terrain edge.
- **Observation**: Proprioceptive only (joint angles, velocities, torso orientation). No height-field sensing — the agent must handle the slope through proprioceptive feedback alone.
- **Key morphological demands**: This is the most physically demanding task. The -10 degree slope creates gravitational forces that pull the agent sideways/downhill. Requires: strong limbs for climbing/braking, low center of gravity for slope stability, body proportions that resist toppling on inclines, and the ability to maintain upright posture (standing bonus is active). Morphologies that excel here likely have robust, powerful builds rather than lightweight speed-optimized ones.""",
}

NEW_TASK_DESCRIPTIONS = {
    "bump": """### New Task: Bumpy Terrain Traversal
- **Environment**: Flat base terrain (50x20 units) densely covered with 250 small bumps. Boundary walls on both sides. Modules: Agent + Terrain + Objects.
- **Bumps**: Each bump has length 0.8-1.6 units, width 0.8-1.6 units, height 0.1-0.25 units. They are densely scattered across the terrain with a 5-unit buffer at the start. Unlike the obstacle task which has fewer but larger obstacles, this has MANY small terrain irregularities.
- **Goal**: Move forward (positive x-direction) as fast as possible over the bumpy terrain.
- **Reward**: `r = 1.0 * x_velocity`. Same as locomotion — pure forward speed. No control cost or standing bonus.
- **Termination**: Episode ends if torso height drops below 50% of initial height, or if agent contacts terrain edge.
- **Observation**: Proprioceptive + 2D height-field observation (sensing grid: 1 unit behind, 4 ahead, 4 left, 4 right). Same observation structure as the obstacle task.
- **Key morphological demands**: This task is a hybrid between locomotion (forward speed) and obstacle (terrain awareness). The bumps are small but extremely dense — the agent cannot avoid them, it must traverse THROUGH them. This favors morphologies with: good shock absorption, stable foothold on uneven surfaces, adaptive gait that handles continuous small perturbations, and robust balance under persistent terrain disturbances. Unlike obstacles where you navigate AROUND, here you must plow THROUGH.""",

    "push_box_incline": """### New Task: Push Box on Incline
- **Environment**: Curved slope terrain (40x20 units) at -10 degree angle, with a 2kg movable box placed at position (-35, 0) and a goal zone at position (35, 0). No boundary walls. Modules: Agent + Terrain + Objects.
- **Goal**: Two-phase manipulation task: (1) Navigate to the box and make contact, then (2) push the box from its start position to the goal position, all while on a -10 degree slope.
- **Reward**: `r = reach_reward + push_reward - ctrl_cost` where:
  - `reach_reward = (agent_obj_distance_before - agent_obj_distance_after) * 100` — rewards getting closer to the box. A +10 bonus is given when the agent first touches the box.
  - `push_reward = (obj_goal_distance_before - obj_goal_distance_after) * 100` — rewards pushing the box toward the goal (only active after the agent has reached the box). A +10 bonus is given when the box reaches the goal.
  - `ctrl_cost` is minimal (weight ~0.0).
- **Termination**: Episode ends if torso height drops below 20% of initial height (same strict threshold as incline).
- **Observation**: Proprioceptive only. No height-field sensing.
- **Key morphological demands**: This is fundamentally different from all locomotion tasks — it requires MANIPULATION on a slope. The agent must: navigate to a specific position (not just forward), exert pushing force on a 2kg object, maintain stability while applying force on a -10 degree slope, and control its body precisely for object interaction rather than pure speed. Favors morphologies with: strong, stable builds (like incline), ability to generate directed force, body shape suitable for pushing objects, and precise locomotion control.""",
}

# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert in evolutionary robotics, morphology-task co-design, and \
reinforcement learning for articulated robots in physics simulation (MuJoCo).

You understand deeply how body structure — number and arrangement of limbs, body proportions, \
joint properties (damping, stiffness, range of motion), mass distribution, center of gravity, \
and overall body topology — affects an agent's ability to perform different locomotion and \
manipulation tasks.

You will analyze empirical performance data from morphology clusters trained on known tasks, \
then predict which morphology clusters would perform best on a new unseen task. Your predictions \
must be grounded in physical reasoning about how morphological traits create advantages or \
disadvantages for specific task demands.

Important: The morphology clusters were created by training a Variational Autoencoder (VAE) on \
morphology descriptors of 50,000 procedurally generated robots, then clustering the latent space \
with K-means (K=20). Each cluster therefore contains morphologically SIMILAR robots — they share \
structural traits like limb count, body proportions, and joint configurations. When a cluster \
performs well on a task, it indicates that the shared morphological traits of that cluster are \
well-suited to that task's demands."""


def build_performance_table_text(perf_data):
    """Build formatted performance table text from best-performer data."""
    # Organize by cluster
    clusters = sorted(set(d["cluster"] for d in perf_data))
    tasks = sorted(set(d["task"] for d in perf_data))

    # Build lookup
    lookup = {}
    for d in perf_data:
        lookup[(d["cluster"], d["task"])] = d["reward"]

    # Raw reward table
    lines = ["#### Raw Reward Scores (higher is better)"]
    header = f"| {'Cluster':>8} |"
    for t in tasks:
        header += f" {t:>14} |"
    lines.append(header)
    lines.append("|" + "----------|" * (1 + len(tasks)))
    for c in clusters:
        row = f"| {c:>8} |"
        for t in tasks:
            val = lookup.get((c, t))
            row += f" {val:>14.1f} |" if val is not None else f" {'N/A':>14} |"
        lines.append(row)
    raw_table = "\n".join(lines)

    # Normalized table [0,1]
    norm_lookup = {}
    for t in tasks:
        vals = [lookup[(c, t)] for c in clusters if (c, t) in lookup]
        if vals:
            min_v, max_v = min(vals), max(vals)
            for c in clusters:
                if (c, t) in lookup:
                    if max_v > min_v:
                        norm_lookup[(c, t)] = (lookup[(c, t)] - min_v) / (max_v - min_v)
                    else:
                        norm_lookup[(c, t)] = 1.0

    lines = ["\n#### Normalized Scores [0.0 = worst, 1.0 = best within each task]"]
    header = f"| {'Cluster':>8} |"
    for t in tasks:
        header += f" {t:>14} |"
    lines.append(header)
    lines.append("|" + "----------|" * (1 + len(tasks)))
    for c in clusters:
        row = f"| {c:>8} |"
        for t in tasks:
            val = norm_lookup.get((c, t))
            row += f" {val:>14.3f} |" if val is not None else f" {'N/A':>14} |"
        lines.append(row)
    norm_table = "\n".join(lines)

    # Ranking per task
    lines = ["\n#### Rank per Task (1 = best)"]
    header = f"| {'Cluster':>8} |"
    for t in tasks:
        header += f" {t:>14} |"
    lines.append(header)
    lines.append("|" + "----------|" * (1 + len(tasks)))
    for t in tasks:
        sorted_clusters = sorted(clusters, key=lambda c: lookup.get((c, t), -1e9), reverse=True)
        for rank, c in enumerate(sorted_clusters, 1):
            norm_lookup.setdefault("rank", {})
    # Build rank lookup
    rank_lookup = {}
    for t in tasks:
        sorted_clusters = sorted(clusters, key=lambda c: lookup.get((c, t), -1e9), reverse=True)
        for rank, c in enumerate(sorted_clusters, 1):
            rank_lookup[(c, t)] = rank
    lines = ["\n#### Rank per Task (1 = best)"]
    header = f"| {'Cluster':>8} |"
    for t in tasks:
        header += f" {t:>14} |"
    lines.append(header)
    lines.append("|" + "----------|" * (1 + len(tasks)))
    for c in clusters:
        row = f"| {c:>8} |"
        for t in tasks:
            rank = rank_lookup.get((c, t))
            row += f" {rank:>14} |" if rank is not None else f" {'N/A':>14} |"
        lines.append(row)
    rank_table = "\n".join(lines)

    # Performance profile narrative
    narratives = []
    for c in clusters:
        scores = {t: lookup.get((c, t)) for t in tasks}
        best_task = max(scores, key=lambda t: scores[t] if scores[t] else -1e9)
        worst_task = min(scores, key=lambda t: scores[t] if scores[t] else 1e9)
        narratives.append(
            f"- **Cluster {c}**: Best at {best_task} ({scores[best_task]:.0f}), "
            f"worst at {worst_task} ({scores[worst_task]:.0f}). "
            f"Ratio best/worst = {scores[best_task]/scores[worst_task]:.1f}x."
        )

    # Cross-cluster comparison
    for t in tasks:
        best_c = max(clusters, key=lambda c: lookup.get((c, t), -1e9))
        second_c = sorted(clusters, key=lambda c: lookup.get((c, t), -1e9), reverse=True)[1]
        ratio = lookup[(best_c, t)] / lookup[(second_c, t)] if lookup.get((second_c, t)) else 0
        narratives.append(
            f"- On **{t}**: Cluster {best_c} leads with {lookup[(best_c, t)]:.0f} "
            f"({ratio:.2f}x over runner-up Cluster {second_c} at {lookup[(second_c, t)]:.0f})."
        )

    profile = "\n#### Performance Profile Analysis\n" + "\n".join(narratives)

    return raw_table + "\n" + norm_table + "\n" + rank_table + "\n" + profile


def build_user_prompt(perf_data, new_task_key, custom_description=None):
    """Compose the full user prompt."""
    sections = []

    # Section 1: Background
    sections.append("""## Background: LOKI Morphology Co-Design

We are studying how different robot body structures (morphologies) perform across different \
physical tasks. Our experimental setup:

1. **Morphology Generation**: 50,000 robot morphologies were procedurally generated with \
varying numbers of limbs (2-10), limb lengths, joint configurations (hinge joints with varying \
damping, stiffness, and range of motion), body segment sizes, and mass distributions. Each \
morphology is a unique articulated robot body.

2. **VAE Clustering**: A Variational Autoencoder (VAE) was trained on numerical descriptors \
of these morphologies (encoding body topology, limb properties, joint parameters). The VAE \
latent space was then clustered using K-means with K=20, producing 20 morphology clusters. \
Each cluster contains hundreds to thousands of morphologically similar robots that share \
structural characteristics.

3. **LOKI Co-Design Training**: For each (cluster, task) pair, we ran the LOKI co-design \
algorithm: starting with 20 randomly sampled morphologies from the cluster, training a shared \
PPO policy for 1×10^8 environment steps, periodically dropping the worst-performing morphologies \
and replacing them with better ones sampled from the same cluster. The final best reward \
achieved by any agent in the cluster represents that cluster's "affinity" for the task.

4. **Current Scope**: We have completed training for 3 clusters (IDs: 0, 2, 18) across 3 \
known tasks. Your job is to analyze these results and predict performance on a new unseen task.""")

    # Section 2: Known task descriptions
    sections.append("\n## Known Task Descriptions\n")
    for task_name in ["locomotion", "obstacle", "incline"]:
        sections.append(KNOWN_TASK_DESCRIPTIONS[task_name])

    # Section 3: Performance table
    sections.append("\n## Empirical Performance Results\n")
    sections.append(
        "The following table shows the best reward achieved by the top-performing agent "
        "in each cluster after full LOKI co-design training (1×10^8 environment steps). "
        "Higher reward means better performance.\n"
    )
    sections.append(build_performance_table_text(perf_data))

    # Section 4: New task + request
    sections.append("\n## New Unseen Task\n")
    if custom_description:
        sections.append(f"### Task: Custom\n{custom_description}")
    elif new_task_key in NEW_TASK_DESCRIPTIONS:
        sections.append(NEW_TASK_DESCRIPTIONS[new_task_key])
    else:
        raise ValueError(f"Unknown task: {new_task_key}")

    sections.append("""
## Your Analysis Request

Based on the performance patterns across the 3 known tasks, predict how each of the 3 clusters \
(0, 2, 18) would perform on the new unseen task described above.

Please provide your analysis in the following structure:

1. **Cross-Task Pattern Analysis**: What patterns do you observe in how the clusters perform \
across locomotion, obstacle, and incline? What does each cluster's performance profile suggest \
about the morphological traits of robots in that cluster?

2. **New Task Demand Analysis**: What specific morphological traits does the new task demand? \
Which of the known tasks is it most similar to, and in what ways does it differ?

3. **Cluster Recommendations**: For each cluster (0, 2, 18), predict its relative performance \
on the new task. Rank them from best to worst, with reasoning grounded in physical analysis \
of how morphological traits transfer between tasks.

4. **Confidence and Caveats**: How confident are you in your predictions? What factors could \
cause your predictions to be wrong?

**IMPORTANT**: Output your response as a JSON object with exactly this schema:
```json
{
  "cross_task_analysis": "Your analysis of patterns across known tasks...",
  "new_task_analysis": "What the new task demands morphologically...",
  "recommendations": [
    {
      "cluster_id": <int>,
      "predicted_rank": <int 1-3>,
      "predicted_performance": "<high|medium|low>",
      "confidence": "<high|medium|low>",
      "reasoning": "Why this cluster would perform at this level..."
    }
  ],
  "key_transfer_insights": "What cross-task correlations informed your predictions...",
  "caveats": "Limitations and potential sources of error..."
}
```

Wrap the JSON in a markdown code block (```json ... ```). Before the JSON, you may include \
your detailed reasoning as free text.""")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# API call and response parsing
# ---------------------------------------------------------------------------

def call_claude(system_prompt, user_prompt, model="claude-sonnet-4-6", max_tokens=4096,
                temperature=0.3):
    """Call Claude API and return the response."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message


def extract_json_from_response(text):
    """Extract JSON object from Claude's response, handling markdown code blocks."""
    # Try to find JSON in a code block
    pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Fallback: try to find a raw JSON object
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-guided morphology cluster recommendation"
    )
    parser.add_argument(
        "--new_task",
        type=str,
        required=True,
        choices=["bump", "push_box_incline", "custom"],
        help="New task to predict cluster performance for",
    )
    parser.add_argument(
        "--custom_description",
        type=str,
        default=None,
        help="Free-text task description (required when --new_task custom)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens for Claude response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for Claude (lower = more deterministic)",
    )
    parser.add_argument(
        "--perf_data",
        type=str,
        default=_BEST_PERFORMERS_PATH,
        help="Path to best_performers_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full prompt and response",
    )
    args = parser.parse_args()

    if args.new_task == "custom" and not args.custom_description:
        parser.error("--custom_description is required when --new_task is 'custom'")

    # Load performance data
    print(f"Loading performance data from {args.perf_data}")
    with open(args.perf_data) as f:
        perf_data = json.load(f)
    print(f"  Loaded {len(perf_data)} entries")

    # Build prompts
    user_prompt = build_user_prompt(perf_data, args.new_task, args.custom_description)
    system_prompt = SYSTEM_PROMPT

    if args.verbose:
        print("\n" + "=" * 80)
        print("SYSTEM PROMPT:")
        print("=" * 80)
        print(system_prompt)
        print("\n" + "=" * 80)
        print("USER PROMPT:")
        print("=" * 80)
        print(user_prompt)
        print("=" * 80)

    # Call API
    print(f"\nCalling {args.model} (max_tokens={args.max_tokens}, temperature={args.temperature})...")
    message = call_claude(
        system_prompt,
        user_prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    response_text = message.content[0].text
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
    }
    print(f"  Response: {usage['input_tokens']} input tokens, {usage['output_tokens']} output tokens")

    if args.verbose:
        print("\n" + "=" * 80)
        print("FULL RESPONSE:")
        print("=" * 80)
        print(response_text)
        print("=" * 80)

    # Parse JSON from response
    parsed_json = extract_json_from_response(response_text)
    if parsed_json is None:
        print("\nWARNING: Could not parse JSON from response. Saving raw text only.")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_label = args.new_task if args.new_task != "custom" else "custom"
    output_path = os.path.join(
        args.output_dir, f"recommendation_{task_label}_{timestamp}.json"
    )

    result = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "new_task": args.new_task,
            "custom_description": args.custom_description,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "usage": usage,
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
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"  CLUSTER RECOMMENDATIONS FOR: {args.new_task.upper()}")
    print("=" * 70)

    if parsed_json:
        if "cross_task_analysis" in parsed_json:
            print(f"\nCross-Task Analysis:\n  {parsed_json['cross_task_analysis'][:200]}...")

        if "recommendations" in parsed_json:
            print("\nPredicted Rankings:")
            recs = sorted(parsed_json["recommendations"], key=lambda r: r.get("predicted_rank", 99))
            for rec in recs:
                print(
                    f"  #{rec.get('predicted_rank', '?')}. Cluster {rec.get('cluster_id', '?')} "
                    f"— {rec.get('predicted_performance', '?')} performance "
                    f"(confidence: {rec.get('confidence', '?')})"
                )
                reasoning = rec.get("reasoning", "")
                if reasoning:
                    # Print first 150 chars of reasoning
                    short = reasoning[:150].replace("\n", " ")
                    print(f"      {short}{'...' if len(reasoning) > 150 else ''}")

        if "caveats" in parsed_json:
            print(f"\nCaveats:\n  {parsed_json['caveats'][:200]}...")
    else:
        # Print raw response excerpt
        print("\n(Could not parse structured JSON — showing raw response excerpt)")
        print(response_text[:500])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
