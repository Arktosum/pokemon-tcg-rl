import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from env import PTCGEnv
from eval_gauntlet import RandomAgent

def calc_live_bf():
    env = PTCGEnv()
    agent = RandomAgent()
    total_options = 0
    total_steps = 0
    
    for i in range(10):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            mask = obs["action_mask"]
            num_opts = int(np.sum(mask))
            if num_opts > 0:
                total_options += num_opts
                total_steps += 1
                
            action = agent.act(obs)
            try:
                obs, reward, is_done, _, _ = env.step(action)
            except:
                break
            done = is_done or env.is_done
            step += 1
            
    print(f"Sampled {total_steps} steps from live environment.")
    print(f"Average Branching Factor: {total_options/total_steps if total_steps>0 else 0:.2f}")

if __name__ == "__main__":
    calc_live_bf()
