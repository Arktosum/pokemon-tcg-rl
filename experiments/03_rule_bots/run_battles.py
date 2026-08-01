"""
run_battles.py — Local battle test harness for rule-based bots.
TITAN V5.0 Rule-Based Bot Curriculum.

Runs 3 matches via kaggle_environments cabt engine:
1. GreedyBot vs RandomBot
2. TacticalBot vs RandomBot
3. TacticalBot vs GreedyBot

Saves timestamped replay JSONs and prints a summary table.
"""

import os
import sys
import json
import shutil
import traceback
from datetime import datetime
from pathlib import Path

# Ensure deck.csv exists in this directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
deck_dst = os.path.join(_this_dir, "deck.csv")
deck_src = os.path.join(_this_dir, "..", "01_baseline", "agent", "deck.csv")
if not os.path.exists(deck_dst) and os.path.exists(deck_src):
    shutil.copy2(deck_src, deck_dst)
    print(f"Copied deck.csv from baseline to {deck_dst}")

# Also ensure logs directory exists
os.makedirs(os.path.join(_this_dir, "logs"), exist_ok=True)

# Pre-add the baseline agent dir (containing the 'cg' package) to sys.path
# so that kaggle_environments can import it when exec'ing our bots in debug mode.
_agent_dir = os.path.abspath(os.path.join(_this_dir, "..", "01_baseline", "agent"))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from kaggle_environments import make

# Agent paths
RANDOM_AGENT = os.path.join(_this_dir, "..", "01_baseline", "agent", "main.py")
GREEDY_AGENT = os.path.join(_this_dir, "greedy_bot.py")
TACTICAL_AGENT = os.path.join(_this_dir, "tactical_bot.py")


def run_match(env, agent1_path, agent2_path, agent1_name, agent2_name):
    """Run a single match and return the result dict."""
    print(f"\n{'='*60}")
    print(f"  {agent1_name} vs {agent2_name}")
    print(f"{'='*60}")

    try:
        steps = env.run([agent1_path, agent2_path])

        # Get rewards from the last step
        last_step = steps[-1] if steps else []
        rewards = []
        statuses = []
        for agent_data in last_step:
            rewards.append(agent_data.get('reward', 0))
            statuses.append(agent_data.get('status', 'UNKNOWN'))

        r1, r2 = rewards[0] if len(rewards) > 0 else 0, rewards[1] if len(rewards) > 1 else 0
        s1, s2 = statuses[0] if len(statuses) > 0 else '?', statuses[1] if len(statuses) > 1 else '?'

        if r1 > r2:
            result = f"{agent1_name} WINS"
        elif r2 > r1:
            result = f"{agent2_name} WINS"
        else:
            result = "DRAW"

        print(f"  Result: {result}")
        print(f"  Rewards: {agent1_name}={r1}, {agent2_name}={r2}")
        print(f"  Statuses: {agent1_name}={s1}, {agent2_name}={s2}")
        print(f"  Steps: {len(steps)}")

        # Save replay
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_filename = f"replay_{agent1_name}_vs_{agent2_name}_{ts}.json"
        replay_path = os.path.join(_this_dir, "logs", replay_filename)
        with open(replay_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(env.toJSON()))
        print(f"  Replay saved: {replay_path}")

        return {
            "match": f"{agent1_name} vs {agent2_name}",
            "result": result,
            "rewards": (r1, r2),
            "statuses": (s1, s2),
            "steps": len(steps),
            "replay": replay_filename,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {
            "match": f"{agent1_name} vs {agent2_name}",
            "result": "ERROR",
            "error": str(e),
        }


def main():
    print("TITAN V5.0 Rule-Based Bot Curriculum — Battle Tests")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nCreating cabt environment...")
    env = make("cabt", debug=True)
    print("Environment created successfully.")

    results = []

    # Match 1: GreedyBot vs RandomBot
    results.append(run_match(env, GREEDY_AGENT, RANDOM_AGENT, "GreedyBot", "RandomBot"))

    # Match 2: TacticalBot vs RandomBot
    results.append(run_match(env, TACTICAL_AGENT, RANDOM_AGENT, "TacticalBot", "RandomBot"))

    # Match 3: TacticalBot vs GreedyBot
    results.append(run_match(env, TACTICAL_AGENT, GREEDY_AGENT, "TacticalBot", "GreedyBot"))

    # Print summary
    print(f"\n{'='*60}")
    print("  BATTLE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Match':<30} {'Result':<20} {'Steps':<10}")
    print(f"{'-'*60}")
    for r in results:
        match = r["match"]
        result = r["result"]
        steps = r.get("steps", "N/A")
        print(f"{match:<30} {result:<20} {steps:<10}")

    # Save summary to log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(_this_dir, "logs", f"battle_summary_{ts}.log")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"TITAN V5.0 Rule-Based Bot Curriculum — Battle Tests\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for r in results:
            f.write(f"Match: {r['match']}\n")
            f.write(f"  Result: {r['result']}\n")
            if 'rewards' in r:
                f.write(f"  Rewards: {r['rewards']}\n")
            if 'statuses' in r:
                f.write(f"  Statuses: {r['statuses']}\n")
            f.write(f"  Steps: {r.get('steps', 'N/A')}\n")
            f.write(f"  Replay: {r.get('replay', 'N/A')}\n\n")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
