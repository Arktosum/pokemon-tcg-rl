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
            with open(replay_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                
            rewards = data.get("rewards", [])
            if len(rewards) < 2:
                continue
                
            r0 = float(rewards[0] or 0)
            r1 = float(rewards[1] or 0)
            
            if r0 > r1:
                winner_idx = 0
            elif r1 > r0:
                winner_idx = 1
            else:
                continue # Draw, skip
                
            steps = data.get("steps", [])
            for i, step_list in enumerate(steps):
                if len(step_list) <= winner_idx:
                    continue
                    
                winner_step = step_list[winner_idx]
                if winner_step is None or "observation" not in winner_step or "action" not in winner_step:
                    continue
                    
                action = winner_step["action"]
                if not action or not isinstance(action, list):
                    continue
                    
                # Skip step 0 (deck submission, length 60)
                if len(action) > 5:
                    continue
                    
                obs_dict = winner_step["observation"]
                if "current" not in obs_dict:
                    continue # Waiting for opponent
                    
                # Process observation to (120,) vector
                try:
                    obs = env._process_obs(obs_dict)
                    state_vec = obs["obs"]
                    action_mask = obs["action_mask"]
                except Exception as e:
                    continue
                    
                action_int = action[0]
                
                # Verify action is within valid bounds
                if action_int >= 500 or action_int < 0:
                    continue
                    
                # Verify action is actually masked as valid
                if action_mask[action_int] == 1:
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
