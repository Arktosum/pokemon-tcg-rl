import os
import sys
import json

# Adjust sys.path to import tournament
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../03_rule_based_eval/evaluation"))
from tournament import run_tournament

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TITAN_AGENT = os.path.join(BASE_DIR, "titan_agent.py")

OPPONENTS = {
    "random_baseline": os.path.join(BASE_DIR, "../01_baseline/agent/main.py"),
    "dragapult":       os.path.join(BASE_DIR, "../03_rule_based_eval/agents/dragapult.py"),
    "abomasnow":       os.path.join(BASE_DIR, "../03_rule_based_eval/agents/abomasnow.py"),
}

if __name__ == "__main__":
    all_results = {}
    NUM_GAMES = 100

    print(f"Starting 100-game BC Final Benchmark against Rule-Based Agents\n")

    for name, agent_path in OPPONENTS.items():
        print(f"\n{'='*50}")
        print(f"  TITAN BC vs {name.upper()} ({NUM_GAMES} games)")
        print(f"{'='*50}")
        print(f"  Agent 1 (Titan): {TITAN_AGENT}")
        print(f"  Agent 2 (Opponent): {agent_path}")

        # Run tournament
        summary = run_tournament(TITAN_AGENT, agent_path, num_games=NUM_GAMES, num_workers=4)
        all_results[name] = summary

        print(f"\n--- {name.upper()} RESULTS ---")
        print(json.dumps(summary, indent=4))

    print("\n\n=== ROUND-ROBIN SUMMARY ===")
    print(f"{'Opponent':<20} {'Titan Wins':>10} {'Opponent Wins':>15} {'Draws':>7} {'Avg Turns':>10} {'Errors':>8}")
    print("-" * 75)
    for name, res in all_results.items():
        print(f"{name:<20} {res['agent1_wins']:>10} {res['agent2_wins']:>15} {res['draws']:>7} {res['avg_turns']:>10.1f} {res['total_errors']:>8}")
    print()
