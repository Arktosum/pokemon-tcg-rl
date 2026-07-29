import sys
import os
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.league_env import LeagueEnv

def run_tests():
    print("Initializing LeagueEnv...")
    env = LeagueEnv()
    
    print("\n--- Running 10 Episodes ---")
    for ep in range(1, 11):
        obs, info = env.reset()
        
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0
        
        try:
            next_obs, reward, done, truncated, info = env.step(action)
            status = "Success"
        except Exception as e:
            status = f"Failed: {e}"
            
        print(f"Episode {ep:02d} | Selected Opponent: {env.current_opponent_name:<25} | Main Agent Player Index: {env.agent_player_idx} | Step Status: {status}")
        
if __name__ == "__main__":
    run_tests()
