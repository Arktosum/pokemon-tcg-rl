import torch
import numpy as np
import os
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import PokemonActorCritic
from env import PTCGEnv
from replay_parser import parse_replay

def sanity_check():
    device = torch.device("cpu")
    model = PokemonActorCritic(num_layers=2)
    model.eval()
    
    ckpt = torch.load("TOP_ELO_BC_MODEL_FINAL.pt", map_location=device)
    model.load_state_dict(ckpt)
    
    # We will pick 2 distinct replay files that we know have valid steps
    replays = [
        "data/replays/episode-87962231-replay.json",
        "data/replays/episode-87962768-replay.json"
    ]
    
    for rp in replays:
        if not os.path.exists(rp):
            continue
        gen = parse_replay(rp)
        try:
            state_vec, action, _ = next(gen)
        except StopIteration:
            continue
            
        print(f"\n--- Board State from {rp} ---")
        print(f"Features [0:10]: {state_vec[:10]}")
        print(f"Features [10:20]: {state_vec[10:20]}")
        
        # Add batch dim
        s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            policy_probs, _ = model(s_tensor)
            
        p = policy_probs.squeeze(0).numpy()
        top_3 = np.argsort(p)[-3:][::-1]
        
        print(f"Top 3 Model Actions:")
        for idx in top_3:
            print(f"  Action {idx:3d}: {p[idx]:.4f} probability")
            
        print(f"Actual human action taken in replay: {action}")

if __name__ == "__main__":
    sanity_check()
