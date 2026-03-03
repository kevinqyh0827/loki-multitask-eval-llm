#!/usr/bin/env python3
"""
Interactive REPL for LLM-guided morphology cluster recommendation.

Loads performance data once, then allows iterative querying with different
task descriptions. Saves conversation history.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python tools/llm_cluster_recommender_interactive.py
"""

import json
import os
import sys
from datetime import datetime

# Reuse core logic from the main tool
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from llm_cluster_recommender import (
    SYSTEM_PROMPT,
    NEW_TASK_DESCRIPTIONS,
    _BEST_PERFORMERS_PATH,
    _DEFAULT_OUTPUT_DIR,
    build_user_prompt,
    call_claude,
    extract_json_from_response,
)


def print_recommendations(parsed_json):
    """Print parsed recommendations in a readable format."""
    if not parsed_json:
        print("  (No structured JSON parsed)")
        return

    if "cross_task_analysis" in parsed_json:
        print(f"\n  Cross-Task Analysis:")
        for line in parsed_json["cross_task_analysis"].split(". "):
            if line.strip():
                print(f"    - {line.strip()}.")

    if "new_task_analysis" in parsed_json:
        print(f"\n  New Task Analysis:")
        for line in parsed_json["new_task_analysis"].split(". "):
            if line.strip():
                print(f"    - {line.strip()}.")

    if "recommendations" in parsed_json:
        print("\n  Predicted Rankings:")
        recs = sorted(parsed_json["recommendations"], key=lambda r: r.get("predicted_rank", 99))
        for rec in recs:
            print(
                f"    #{rec.get('predicted_rank', '?')}. Cluster {rec.get('cluster_id', '?')} "
                f"[{rec.get('predicted_performance', '?')}] "
                f"(confidence: {rec.get('confidence', '?')})"
            )
            if rec.get("reasoning"):
                print(f"        Reasoning: {rec['reasoning']}")

    if "key_transfer_insights" in parsed_json:
        print(f"\n  Key Insights: {parsed_json['key_transfer_insights']}")

    if "caveats" in parsed_json:
        print(f"\n  Caveats: {parsed_json['caveats']}")


def main():
    model = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
    perf_path = os.environ.get("PERF_DATA_PATH", _BEST_PERFORMERS_PATH)

    print("=" * 70)
    print("  LOKI Cluster Recommender — Interactive Mode")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Performance data: {perf_path}")
    print()

    # Load performance data
    with open(perf_path) as f:
        perf_data = json.load(f)
    print(f"  Loaded {len(perf_data)} performance entries")

    # Show available built-in tasks
    print(f"\n  Built-in new tasks: {', '.join(NEW_TASK_DESCRIPTIONS.keys())}")
    print("  Or type any free-text task description.")
    print("  Type 'quit' or 'exit' to stop. Type 'save' to save conversation history.\n")

    conversation_history = []
    output_dir = _DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    while True:
        try:
            user_input = input("New task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "save":
            save_path = os.path.join(
                output_dir,
                f"interactive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
            with open(save_path, "w") as f:
                json.dump(conversation_history, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(conversation_history)} queries to {save_path}")
            continue

        # Determine if it's a built-in task or custom
        if user_input.lower() in NEW_TASK_DESCRIPTIONS:
            new_task_key = user_input.lower()
            custom_desc = None
            task_label = new_task_key
        else:
            new_task_key = "custom"
            custom_desc = user_input
            task_label = "custom"

        # Build prompt and call API
        try:
            user_prompt = build_user_prompt(perf_data, new_task_key, custom_desc)
            print(f"\n  Querying {model}...")
            message = call_claude(
                SYSTEM_PROMPT, user_prompt, model=model,
                max_tokens=4096, temperature=temperature,
            )
            response_text = message.content[0].text
            parsed = extract_json_from_response(response_text)

            usage = {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            }
            print(f"  ({usage['input_tokens']} in / {usage['output_tokens']} out tokens)")

            # Show results
            print("\n" + "-" * 60)
            print(f"  RECOMMENDATION FOR: {task_label.upper()}")
            print("-" * 60)
            print_recommendations(parsed)
            print("-" * 60 + "\n")

            # Save to history
            conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "task_input": user_input,
                "task_key": new_task_key,
                "usage": usage,
                "parsed_recommendation": parsed,
                "raw_response": response_text,
            })

        except anthropic.APIError as e:
            print(f"  API Error: {e}")
        except Exception as e:
            print(f"  Error: {e}")

    # Auto-save on exit if there's history
    if conversation_history:
        save_path = os.path.join(
            output_dir,
            f"interactive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(save_path, "w") as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)
        print(f"\nSession saved to {save_path}")


if __name__ == "__main__":
    import anthropic as anthropic_module
    main()
