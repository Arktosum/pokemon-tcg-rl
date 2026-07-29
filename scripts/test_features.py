import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from replay_parser import parse_replay

REPLAY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "replays")
replay_files = [f for f in os.listdir(REPLAY_DIR) if f.endswith('.json')]

if not replay_files:
    print("No replay files found.")
    sys.exit(0)

# Pre-flight Check: Process 5 replays fully and report stats
print("=== PRE-FLIGHT CHECK ===")
total_steps = 0
kept_steps = 0
dropped_opponent_turn = 0
dropped_no_current = 0
dropped_bad_action = 0

for i in range(min(5, len(replay_files))):
    filepath = os.path.join(REPLAY_DIR, replay_files[i])
    print(f"\nProcessing {replay_files[i]}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rewards = data.get('rewards', [0, 0])
    winner_idx = 0 if rewards[0] > rewards[1] else 1
    steps = data.get('steps', [])
    
    # Just to get the raw numbers
    for step_group in steps:
        if len(step_group) > winner_idx:
            total_steps += 1
            agent_step = step_group[winner_idx]
            
            raw_action = agent_step.get('action')
            if raw_action is None or len(raw_action) == 0:
                dropped_opponent_turn += 1
                continue
                
            raw_obs = agent_step.get('observation')
            if raw_obs is None or raw_obs.get('current') is None:
                dropped_no_current += 1
                continue
                
            # Extract action safely
            if isinstance(raw_action, list):
                if isinstance(raw_action[0], list):
                    act_val = raw_action[0][0] if len(raw_action[0]) > 0 else 0
                else:
                    act_val = raw_action[0]
            else:
                act_val = raw_action
                
            action = int(act_val)
            
            # Reject out-of-bounds actions rather than wrapping them silently
            if action < 0 or action >= 500:
                dropped_bad_action += 1
                continue
            
            kept_steps += 1
    
    # Actually run parse_replay to get a sample vector
    gen = parse_replay(filepath)
    try:
        sample_state, sample_action, _ = next(gen)
        print(f"Sample State (first 20 elements):\n{sample_state[:20]}")
        print(f"Sample Action: {sample_action}")
    except StopIteration:
        print("No valid steps yielded.")
        
print(f"\n=== PRE-FLIGHT STATS ===")
print(f"Total Steps: {total_steps}")
print(f"Kept Steps: {kept_steps}")
print(f"Dropped (opponent's turn / no action): {dropped_opponent_turn}")
print(f"Dropped (no 'current' state): {dropped_no_current}")
print(f"Dropped (bad action index): {dropped_bad_action}")
print(f"Sum check (Kept + Dropped): {kept_steps + dropped_opponent_turn + dropped_no_current + dropped_bad_action}")
