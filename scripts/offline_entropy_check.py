import json
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ENGINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_submission', 'sample_submission'))
if ENGINE_PATH not in sys.path:
    sys.path.append(ENGINE_PATH)

from src.model import PokemonActorCritic
from src.main import _process_obs

class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            elif isinstance(v, list):
                self.__dict__[k] = [Struct(**item) if isinstance(item, dict) else item for item in v]
            else:
                self.__dict__[k] = v
                
    def __getattr__(self, name):
        return None

def calculate_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0

def run():
    print("Loading model...")
    model = PokemonActorCritic(num_layers=3)
    checkpoint = torch.load("checkpoints/TOP_ELO_BC_MODEL_FINAL.pt", map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("Loading replay...")
    with open("episode-88792394-replay.json", "r") as f:
        data = json.load(f)

    steps = data.get("steps", [])
    print(f"Turn Number | Actual Move Taken | Top Logit Probability | Entropy")
    print("-" * 75)

    agent_id = 1
    
    for turn_number, step in enumerate(steps):
        if not isinstance(step, list) or len(step) <= agent_id:
            continue
            
        agent_data = step[agent_id]
        if agent_data is None:
            continue
            
        obs_dict = agent_data.get("observation", {})
        if not obs_dict:
            continue
            
        action_taken = agent_data.get("action")
        if action_taken is None or (isinstance(action_taken, list) and len(action_taken) == 0):
            continue

        # Wrap in Struct
        obs = Struct(**obs_dict)
        
        try:
            state_vec, mask = _process_obs(obs)
        except Exception as e:
            continue

        with torch.no_grad():
            s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
            policy, _ = model(s_tensor, m_tensor)
            
        p = policy.squeeze(0).numpy()
        valid_actions = np.where(np.array(mask) == 1)[0]
        
        if len(valid_actions) > 0:
            p_valid = p[valid_actions]
            if p_valid.sum() > 0:
                p_valid /= p_valid.sum()
            else:
                p_valid = np.ones_like(p_valid) / len(p_valid)
                
            entropy = calculate_entropy(p_valid)
            top_prob = float(np.max(p_valid))
        else:
            entropy = 0.0
            top_prob = 0.0

        if isinstance(action_taken, list):
            action_taken_str = str(action_taken[0]) if len(action_taken) > 0 else "None"
        else:
            action_taken_str = str(action_taken)
            
        print(f"{turn_number:^11} | {action_taken_str:^17} | {top_prob:^21.4f} | {entropy:^7.4f}")

if __name__ == '__main__':
    run()
