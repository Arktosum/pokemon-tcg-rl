import time
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from env import PTCGEnv
from model import PokemonActorCritic
from greedy_agent import GreedyAgent

def play_match(env, model, greedy_agent):
    obs, _ = env.reset()
    done = False
    step = 0
    while not done and step < 200:
        state_vec = obs["obs"]
        mask = obs["action_mask"]
        current_player = int(state_vec[2])
        if current_player == 0:
            s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
            with torch.no_grad():
                policy, _ = model(s_tensor, m_tensor)
            p = policy.squeeze(0).numpy()
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) > 0:
                p_valid = p[valid_actions]
                if p_valid.sum() > 0:
                    p_valid /= p_valid.sum()
                    action = int(np.random.choice(valid_actions, p=p_valid))
                else:
                    action = int(np.random.choice(valid_actions))
            else:
                break
        else:
            action = greedy_agent.act(obs)
        try:
            obs, reward, is_done, _, _ = env.step(action)
        except Exception as e:
            # If the engine crashes, treat as a loss
            break
        done = is_done or env.is_done
        step += 1
    winner = env.winner if hasattr(env, 'winner') else -1
    return 1 if winner == 0 else 0

def run_eval():
    env = PTCGEnv()
    model = PokemonActorCritic()
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TITAN_GREEDY_PPO_01.pt")
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # Handle cases where the key might be missing
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model.eval()
    greedy_agent = GreedyAgent()
    
    wins_greedy = 0
    num_games = 1000
    
    print(f"Starting 1000-game offline evaluation of TITAN_TRANSFORMER_LEAGUE_01.pt vs GreedyAgent...")
    for i in range(num_games):
        if i % 100 == 0:
            print(f"Game {i}/{num_games} completed...")
        wins_greedy += play_match(env, model, greedy_agent)
        
    print(f"Evaluation complete.")
    print(f"Win/Loss Count vs Greedy: {wins_greedy} Wins / {num_games - wins_greedy} Losses")

if __name__ == "__main__":
    run_eval()
