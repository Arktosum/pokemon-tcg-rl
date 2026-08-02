import os
import sys
import json

# Adjust sys.path to import tournament
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation"))
from tournament import run_tournament

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.abspath(os.path.join(BASE_DIR, "../01_baseline/agent/main.py"))

AGENTS = {
    "iono":          os.path.join(BASE_DIR, "agents/iono.py"),
    "abomasnow":     os.path.join(BASE_DIR, "agents/abomasnow.py"),
    "probabilistic": os.path.join(BASE_DIR, "agents/probabilistic.py"),
}

if __name__ == "__main__":
    all_results = {}

    for name, agent_path in AGENTS.items():
        print(f"\n{'='*50}")
        print(f"  {name.upper()} vs RANDOM BASELINE (5 games)")
        print(f"{'='*50}")
        print(f"  Agent 1: {agent_path}")
        print(f"  Agent 2: {BASELINE_PATH}")

        summary = run_tournament(agent_path, BASELINE_PATH, num_games=5, num_workers=2)
        all_results[name] = summary

        print(f"\n--- {name.upper()} RESULTS ---")
        print(json.dumps(summary, indent=4))

    print("\n\n=== ROUND-ROBIN SUMMARY ===")
    print(f"{'Agent':<20} {'Wins':>6} {'Losses':>8} {'Draws':>7} {'Avg Turns':>10} {'Errors':>8}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<20} {res['agent1_wins']:>6} {res['agent2_wins']:>8} {res['draws']:>7} {res['avg_turns']:>10.1f} {res['total_errors']:>8}")
    print()
