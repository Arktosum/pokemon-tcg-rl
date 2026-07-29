import glob
import json
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from env import PTCGEnv

def test():
    env = PTCGEnv()
    rp = glob.glob("data/replays/*.json")[0]
    total_options = 0
    total_steps = 0
    with open(rp, 'r') as f:
        data = json.load(f)
        for step_group in data:
            if isinstance(step_group, list):
                for step in step_group:
                    raw_obs = step.get('observation', {})
                    if 'current' in raw_obs:
                        s_dict = env._process_obs(raw_obs)
                        mask = s_dict['action_mask']
                        total_options += int(np.sum(mask))
                        total_steps += 1
                        
    print(f"Sampled game: {rp}")
    print(f"Total options: {total_options}")
    print(f"Total steps: {total_steps}")
    print(f"Avg branching factor: {total_options/total_steps if total_steps>0 else 0}")

if __name__ == "__main__":
    test()
