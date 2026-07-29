import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from model import PokemonActorCritic
import sys
sys.path.append(os.path.dirname(__file__))
from eval_gauntlet import GreedyAgent, AdvancedHeuristicAgent
from env import PTCGEnv

def get_model_action(model, obs_dict, env):
    state_vec = obs_dict["obs"]
    mask = obs_dict["action_mask"]
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

def trace_match(env, p1_model, opponent, opp_name):
    print(f"\n==============================================")
    print(f"TRACING GAME: TOP_ELO_BC_MODEL_FINAL vs {opp_name}")
    print(f"==============================================\n")
    obs, _ = env.reset()
    done = False
    step = 0
    while not done and step < 50:
        state_vec = obs["obs"]
        current_player = int(state_vec[2])
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        
        if current_player == 0:
            action = get_model_action(p1_model, obs, env)
            agent_str = "MODEL"
        else:
            if hasattr(opponent, 'act'):
                try:
                    action = opponent.act(obs, env)
                except TypeError:
                    action = opponent.act(obs)
            else:
                action = 0
            agent_str = opp_name
            
        # Get human readable action
        action_desc = "Unknown"
        if hasattr(env, 'current_obs') and env.current_obs and env.current_obs.select:
            opts = env.current_obs.select.option
            if action < len(opts):
                opt = opts[action]
                # Try to extract something readable
                if hasattr(opt, 'message') and opt.message:
                    action_desc = opt.message
                else:
                    action_desc = str(opt)
        
        print(f"Step {step:02d} | Player: {agent_str}")
        print(f"  Board State summary: Turn={state_vec[0]}, FirstP={state_vec[3]}, Hand={state_vec[5]}, Deck={state_vec[6]}, Prize={state_vec[4]}")
        print(f"  Action Index: {action}")
        print(f"  Action Meaning: {action_desc}")
        print("-" * 40)
        
        try:
            obs, reward, is_done, _, _ = env.step(action)
        except Exception as e:
            print(f"Game crashed with exception: {e}")
            break
            
        done = is_done or env.is_done
        step += 1
        
    winner = env.winner if hasattr(env, 'winner') else -1
    print(f"Game finished. Winner: {winner}")

def main():
    env = PTCGEnv()
    
    model = PokemonActorCritic(num_layers=2)
    model.load_state_dict(torch.load("TOP_ELO_BC_MODEL_FINAL.pt", map_location='cpu'))
    model.eval()
    
    trace_match(env, model, GreedyAgent(), "GreedyAgent")
    trace_match(env, model, AdvancedHeuristicAgent(), "AdvancedHeuristicAgent")

if __name__ == "__main__":
    main()
