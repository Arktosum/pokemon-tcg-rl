import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust sys.path to import tournament
sys.path.insert(0, os.path.join(BASE_DIR, "../03_rule_based_eval/evaluation"))
from tournament import run_tournament

TITAN_AGENT = os.path.join(BASE_DIR, "titan_agent.py")
DRAGAPULT_AGENT = os.path.abspath(os.path.join(BASE_DIR, "../03_rule_based_eval/agents/dragapult.py"))

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  TITAN AGENT vs DRAGAPULT (10 games)")
    print(f"{'='*50}")
    print(f"  Agent 1: {TITAN_AGENT}")
    print(f"  Agent 2: {DRAGAPULT_AGENT}")

    summary = run_tournament(TITAN_AGENT, DRAGAPULT_AGENT, num_games=10, num_workers=4)

    print(f"\n--- TOURNAMENT RESULTS ---")
    print(json.dumps(summary, indent=4))
