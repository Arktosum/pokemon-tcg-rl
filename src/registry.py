import os
import json
from datetime import datetime

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "registry.json")

def register_model(name, filepath, win_random, win_greedy):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
        
    registry[name] = {
        "filepath": filepath,
        "WinRate_Random": win_random,
        "WinRate_Greedy": win_greedy,
        "Training_Timestamp": datetime.now().isoformat()
    }
    
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)
