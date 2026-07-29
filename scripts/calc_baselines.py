import torch
import numpy as np
import os
import glob
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.replay_parser import parse_replay

def calc_baselines():
    print("--- Calculating Majority Baseline ---")
    data_path = "data/bc_train_full.pt"
    if os.path.exists(data_path):
        data = torch.load(data_path)
        actions = data['actions'].numpy()
        unique, counts = np.unique(actions, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_action = unique[majority_idx]
        majority_count = counts[majority_idx]
        majority_acc = (majority_count / len(actions)) * 100
        print(f"Total training pairs: {len(actions)}")
        print(f"Majority action index: {majority_action}")
        print(f"Majority action frequency: {majority_acc:.2f}%")
    else:
        print("Could not find bc_train_full.pt")
        
    print("\n--- Calculating Average Branching Factor ---")
    # Parse a few replays to get the average number of valid actions
    replay_files = glob.glob("data/replays/*.json")[:50]  # Just use 50 replays for a quick estimate
    
    total_options = 0
    total_steps = 0
    
    for rp in replay_files:
        from src.env import PTCGEnv
        env = PTCGEnv()
        
        with open(rp, 'r') as f:
            for line in f:
                if not line.strip(): continue
                step_group = json.loads(line)
                
                for step in step_group:
                    if 'observation' in step and 'current' in step['observation']:
                        try:
                            s_dict = env._process_obs(step['observation'])
                            mask = s_dict['action_mask']
                            num_opts = int(np.sum(mask))
                            if num_opts > 0:
                                total_options += num_opts
                                total_steps += 1
                        except:
                            pass
                
    if total_steps > 0:
        avg_branching = total_options / total_steps
        print(f"Sampled {total_steps} steps from {len(replay_files)} games.")
        print(f"Average Branching Factor (valid actions per step): {avg_branching:.2f}")

if __name__ == "__main__":
    calc_baselines()
