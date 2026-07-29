import time
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from env import PTCGEnv
from model import PokemonActorCritic
from greedy_agent import GreedyAgent
from advanced_agents import RandomAgent, AdvancedHeuristicAgent

def get_model_action(model, obs, env):
    state_vec = obs["obs"]
    mask = obs["action_mask"]
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
        action = 0
    return action

def play_match(env, p1_model, opponent, opp_is_model=False):
    obs, _ = env.reset()
    done = False
    step = 0
    while not done and step < 200:
        state_vec = obs["obs"]
        current_player = int(state_vec[2])
        if current_player == 0:
            action = get_model_action(p1_model, obs, env)
        else:
            if opp_is_model:
                action = get_model_action(opponent, obs, env)
            else:
                # Some agents expect just obs, some expect (obs, env)
                if hasattr(opponent, 'act'):
                    # Check if it accepts env
                    try:
                        action = opponent.act(obs, env)
                    except TypeError:
                        action = opponent.act(obs)
                else:
                    action = 0
        try:
            obs, reward, is_done, _, _ = env.step(action)
        except Exception as e:
            break
        done = is_done or env.is_done
        step += 1
    winner = env.winner if hasattr(env, 'winner') else -1
    return 1 if winner == 0 else 0

def run_gauntlet():
    env = PTCGEnv()
    
    # Load Main Model
    main_model = PokemonActorCritic()
    final_ckpt = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TOP_ELO_BC_MODEL_FINAL.pt")
    if os.path.exists(final_ckpt):
        checkpoint = torch.load(final_ckpt, map_location='cpu')
        main_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    else:
        print(f"Main Checkpoint not found: {final_ckpt}")
        return
    main_model.eval()
    
    # Load v1 Model
    v1_model = PokemonActorCritic()
    v1_ckpt = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TOP_ELO_BC_MODEL_v1.pt")
    if os.path.exists(v1_ckpt):
        checkpoint_v1 = torch.load(v1_ckpt, map_location='cpu')
        v1_model.load_state_dict(checkpoint_v1.get('model_state_dict', checkpoint_v1))
    else:
        print(f"V1 Checkpoint not found: {v1_ckpt}")
        return
    v1_model.eval()

    opponents = [
        ("RandomAgent", RandomAgent(), False),
        ("GreedyAgent", GreedyAgent(), False),
        ("AdvancedHeuristicAgent", AdvancedHeuristicAgent(), False),
        ("TOP_ELO_BC_MODEL_v1.pt", v1_model, True)
    ]
    
    num_games = 500
    
    print(f"--- THE GAUNTLET EVALUATION ---")
    print(f"Main Agent: TOP_ELO_BC_MODEL_FINAL.pt")
    
    results = {}
    
    for opp_name, opp_agent, opp_is_model in opponents:
        print(f"\nEvaluating vs {opp_name} ({num_games} matches)...")
        wins = 0
        for i in range(num_games):
            if i > 0 and i % 100 == 0:
                print(f"  Game {i}/{num_games} completed...")
            wins += play_match(env, main_model, opp_agent, opp_is_model)
        
        losses = num_games - wins
        results[opp_name] = (wins, losses)
        print(f"Win/Loss Count vs {opp_name}: {wins} Wins / {losses} Losses")
        
    print("\n--- FINAL GAUNTLET RESULTS ---")
    for opp_name, (w, l) in results.items():
        print(f"vs {opp_name}: {w} Wins / {l} Losses ({(w/num_games)*100:.1f}%)")

if __name__ == "__main__":
    run_gauntlet()
