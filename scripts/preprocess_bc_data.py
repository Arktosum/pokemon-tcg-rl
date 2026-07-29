import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
from src.env import PTCGEnv
from src.replay_parser import parse_replay

def process_replays():
    replay_dir = REPO_ROOT / "data" / "replays"
    replays = list(replay_dir.glob("*.json"))
    
    if not replays:
        print("No replays found in data/replays/")
        return
        
    print(f"Found {len(replays)} replays. Starting preprocessing...")
    
    env = PTCGEnv()
    
    all_states = []
    all_actions = []
    
    for replay_path in tqdm(replays):
        try:
            for state_vec, action_int, reward in parse_replay(replay_path):
                all_states.append(state_vec)
                all_actions.append(action_int)
        except Exception as e:
            print(f"Error processing {replay_path}: {e}")
            continue
            
    if not all_states:
        print("No valid state-action pairs extracted.")
        return
        
    print(f"Extracted {len(all_states)} valid state-action pairs.")
    
    # Convert to tensors
    states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32)
    actions_tensor = torch.tensor(np.array(all_actions), dtype=torch.long)
    
    # Shuffle dataset
    indices = torch.randperm(len(states_tensor))
    states_tensor = states_tensor[indices]
    actions_tensor = actions_tensor[indices]
    
    # 80/20 Split
    split_idx = int(0.8 * len(states_tensor))
    train_states = states_tensor[:split_idx]
    train_actions = actions_tensor[:split_idx]
    val_states = states_tensor[split_idx:]
    val_actions = actions_tensor[split_idx:]
    
    print(f"Total pairs: {len(states_tensor)}")
    print(f"Train split size: {len(train_states)}")
    print(f"Val split size: {len(val_states)}")
    
    train_out = REPO_ROOT / "data" / "bc_train_full.pt"
    val_out = REPO_ROOT / "data" / "bc_val_full.pt"
    
    torch.save({"states": train_states, "actions": train_actions}, train_out)
    torch.save({"states": val_states, "actions": val_actions}, val_out)
    
    print(f"Successfully saved train dataset to {train_out} ({os.path.getsize(train_out) / 1024 / 1024:.2f} MB)")
    print(f"Successfully saved val dataset to {val_out} ({os.path.getsize(val_out) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    process_replays()
