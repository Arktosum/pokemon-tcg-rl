import sys
import os
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from env import PTCGEnv
from model import PokemonActorCritic
from greedy_agent import GreedyAgent

# Load Model
model = PokemonActorCritic()
ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "TOP_ELO_BC_MODEL_FINAL.pt")
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
model.eval()

env = PTCGEnv()
greedy = GreedyAgent()

OPTION_TYPES = {
    0: 'NUMBER', 1: 'YES', 2: 'NO', 3: 'CARD', 4: 'TOOL_CARD', 5: 'ENERGY_CARD', 
    6: 'ENERGY', 7: 'PLAY', 8: 'ATTACH', 9: 'EVOLVE', 10: 'ABILITY', 11: 'DISCARD', 
    12: 'RETREAT', 13: 'ATTACK', 14: 'END', 15: 'SKILL', 16: 'SPECIAL_CONDITION'
}

def option_to_str(opt):
    t_name = OPTION_TYPES.get(opt.type, str(opt.type))
    res = f"{t_name}"
    if opt.index is not None: res += f", index: {opt.index}"
    if opt.cardId is not None: res += f", cardId: {opt.cardId}"
    if opt.attackId is not None: res += f", attackId: {opt.attackId}"
    if opt.energyIndex is not None: res += f", energyIndex: {opt.energyIndex}"
    return res

def run_audit(opponent_name, num_games):
    for g in range(num_games):
        print(f"\n==============================================")
        print(f"GAME {g+1} vs {opponent_name}")
        print(f"==============================================\n")
        
        obs, _ = env.reset()
        done = False
        step = 0
        p1_idx = 0
        
        while not done and step < 200:
            state_vec = obs['obs']
            mask = obs['action_mask']
            
            # Simple interpretation of state for log
            turn = int(state_vec[0])
            curr_player = int(state_vec[2])
            p1_active_hp = state_vec[14] * 350.0
            p2_active_hp = state_vec[14 + 19] * 350.0 # roughly
            
            if curr_player == p1_idx:
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
                    
                opt = env.current_obs.select.option[action]
                print(f"[Step {step:03d} | Turn {turn}] BC_MODEL plays Option {action} ({option_to_str(opt)})")
                
            else:
                if opponent_name == "Greedy":
                    action = greedy.act(obs)
                else:
                    from advanced_agents import AdvancedHeuristicAgent
                    adv = AdvancedHeuristicAgent()
                    action = adv.act(obs, env)
                opt = env.current_obs.select.option[action]
                print(f"[Step {step:03d} | Turn {turn}] {opponent_name} plays Option {action} ({option_to_str(opt)})")
                
            obs, reward, is_done, _, _ = env.step(action)
            done = is_done or env.is_done
            step += 1
            
        print(f"Game finished at step {step} with reward {reward}")

import io
orig_stdout = sys.stdout
f_out = open('game_audit_utf8.txt', 'w', encoding='utf-8')
sys.stdout = f_out

run_audit("Greedy", 5)
run_audit("Advanced", 2)

sys.stdout = orig_stdout
f_out.close()
