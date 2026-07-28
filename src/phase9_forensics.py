import time
import numpy as np
import torch
import os
from env import PTCGEnv
from model import PokemonAlphaNet
from puct import PUCTSearch

def fix_markdowns():
    with open("01_JOURNEY_LOG.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add ## to ENTRY if missing
    content = content.replace("ENTRY 006:", "## ENTRY 006:")
    content = content.replace("ENTRY 007:", "## ENTRY 007:")
    content = content.replace("ENTRY 008:", "## ENTRY 008:")
    with open("01_JOURNEY_LOG.md", "w", encoding="utf-8") as f:
        f.write(content)
        
    with open("02_EXPERIMENT_TRACKER.md", "a", encoding="utf-8") as f:
        f.write("\n| `009` | 2026-07-28 12:46 | Phase 9: Forensic Diagnostic | N/A | Loss Autopsy, Tensor Audit, Reward Audit | [ACTIVE LOCK] |\n")

def run_audits():
    env = PTCGEnv()
    model = PokemonAlphaNet()
    if os.path.exists("checkpoints/latest_model.pt"):
        checkpoint = torch.load("checkpoints/latest_model.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    losses = 0
    loss_reasons = []
    
    scalar_min = float('inf')
    scalar_max = float('-inf')
    
    win_reward = None
    loss_reward = None
    
    while losses < 10:
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 500:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            current_player = int(state_vec[2])
            
            # Tensor Audit
            if step > 5:
                # Based on V2: first ~60 slots are Card IDs (categorical), remaining are stats (HP, Damage, etc)
                scalars = state_vec[60:]
                s_min = np.min(scalars)
                s_max = np.max(scalars)
                if s_min < scalar_min: scalar_min = s_min
                if s_max > scalar_max: scalar_max = s_max

            try:
                if current_player == 0:
                    searcher = PUCTSearch(model, num_simulations=10)
                    policy = searcher.search(state_vec, mask, current_player)
                    valid_actions = np.where(mask == 1)[0]
                    if len(valid_actions) > 0:
                        p_valid = policy[valid_actions]
                        if p_valid.sum() > 0:
                            p_valid /= p_valid.sum()
                            action = int(np.random.choice(valid_actions, p=p_valid))
                        else:
                            action = int(np.random.choice(valid_actions))
                    else:
                        action = 0 # Fallback
                else:
                    valid_actions = np.where(mask == 1)[0]
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                    else:
                        action = 0
                        
                obs, reward, is_done, _, info = env.step(action)
                done = is_done or env.is_done
                step += 1
                
                if done:
                    winner = env.winner if hasattr(env, 'winner') else -1
                    if winner == 1: # We lost
                        losses += 1
                        
                        # Infer reason
                        reason = "Opponent Took 6 Prizes"
                        if "reason" in info:
                            reason = info["reason"]
                        elif step > 200: # Turn limit/deck out approximation for simulation
                            reason = "Deck Out / Turn Limit Exceeded"
                        
                        loss_reasons.append(reason)
                        loss_reward = reward
                    elif winner == 0:
                        win_reward = reward
                        
            except Exception as e:
                # If engine throws exception, that's a DQ
                losses += 1
                loss_reasons.append(f"Engine Disqualification / Invalid Action: {e}")
                loss_reward = -1.0 # Standard DQ reward
                break

    print("--- Forensic Audit Results ---")
    print(f"Loss Reasons: {loss_reasons}")
    print(f"Scalar Min: {scalar_min}, Scalar Max: {scalar_max}")
    print(f"Win Reward: {win_reward}, Loss Reward: {loss_reward}")
    
    with open("diagnostic_results.txt", "w", encoding="utf-8") as f:
        f.write(f"{loss_reasons}\n{scalar_min}\n{scalar_max}\n{win_reward}\n{loss_reward}")

if __name__ == "__main__":
    fix_markdowns()
    run_audits()
