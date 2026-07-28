import time
import numpy as np
import torch
from env import PTCGEnv
from model import PokemonActorCritic
import os
from greedy_agent import GreedyAgent

def play_match(env, model, opponent_type):
    obs, _ = env.reset()
    done = False
    step = 0
    greedy_agent = GreedyAgent()
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
            if opponent_type == "random":
                valid_actions = np.where(mask == 1)[0]
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                else:
                    break
            elif opponent_type == "greedy":
                action = greedy_agent.act(obs)
        try:
            obs, reward, is_done, _, _ = env.step(action)
        except:
            break
        done = is_done or env.is_done
        step += 1
    winner = env.winner if hasattr(env, 'winner') else -1
    return 1 if winner == 0 else 0

def run_eval():
    env = PTCGEnv()
    model = PokemonActorCritic()
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "latest_model.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    wins_random = 0
    for i in range(500):
        wins_random += play_match(env, model, "random")
        
    wins_greedy = 0
    for i in range(500):
        wins_greedy += play_match(env, model, "greedy")
        
    rate_random = (wins_random / 500.0) * 100
    rate_greedy = (wins_greedy / 500.0) * 100
    print(f"Win Rate vs Random: {wins_random}/500")
    print(f"Win Rate vs Repaired Greedy: {wins_greedy}/500")
    if rate_random < 95 or rate_greedy < 80:
        print("FAILED CRITERIA")
    else:
        print("PASSED CRITERIA")
        
    try:
        from registry import register_model
        named_ckpt = "TITAN_TRANSFORMER_01.pt"
        named_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", named_ckpt)
        register_model("TITAN_TRANSFORMER_01", named_path, wins_random/500.0, wins_greedy/500.0)
    except ImportError:
        pass

if __name__ == "__main__":
    run_eval()
