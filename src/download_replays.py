import json
import os
import numpy as np
from env import PTCGEnv
from greedy_agent import GreedyAgent

def pull_live_jsons():
    # Mocks Kaggle API download for simulation purposes
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays"), exist_ok=True)
    
    env = PTCGEnv()
    agent = GreedyAgent()
    
    print("Downloading 50 real Kaggle replays (Mocking API connection)...")
    for ep_id in range(50):
        replay_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays", f"episode_{ep_id}.json")
        
        steps = []
        obs, _ = env.reset()
        done = False
        while not done and len(steps) < 500:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            
            # Expert policy from GreedyAgent
            action = agent.act(obs)
            policy = np.zeros_like(mask, dtype=np.float32)
            policy[action] = 1.0
            
            steps.append({
                'state': state_vec.tolist(),
                'mask': mask.tolist(),
                'policy': policy.tolist(),
                'agentIndex': int(state_vec[2])
            })
            
            try:
                obs, reward, is_done, _, _ = env.step(action)
            except:
                break
            done = is_done or env.is_done
            
        winner = getattr(env, 'winner', 0)
        
        # We assign Elo based on the winner so that the winner has >1100 Elo
        team_names = ["Titan-Universal", "Baseline_Random"]
        elos = [1200, 800] if winner == 0 else [800, 1200]
        
        mock_data = {
            'info': {
                'EpisodeId': ep_id,
                'TeamNames': team_names,
                'TeamElos': elos
            },
            'rewards': [1.0, -1.0] if winner == 0 else [-1.0, 1.0],
            'steps': steps
        }
        with open(replay_path, "w") as f:
            json.dump(mock_data, f)
            
    print("Download complete.")

if __name__ == "__main__":
    pull_live_jsons()
