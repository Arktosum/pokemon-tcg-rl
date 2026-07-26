import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)
sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from cg.api import to_observation_class
from src.model.transformer_policy import MyModel, get_encoder_input, get_decoder_input
from src.env.tcg_env import PokemonTCGEnv
from src.model.heuristic_bot import agent as rule_agent

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    # append 0 for the terminal value
    values = values + [0.0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def run_ppo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PPO Training on {device}")
    
    deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    env = PokemonTCGEnv(deck)
    
    model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    clip_ratio = 0.2
    epochs_per_update = 2
    episodes = 2
    
    for episode in range(episodes):
        obs_dict = env.reset()
        
        states_enc, states_dec, actions, log_probs_old, values, rewards = [], [], [], [], [], []
        
        print(f"\n--- Episode {episode + 1} ---")
        step_count = 0
        
        while True:
            if not obs_dict:
                break
                
            your_index = obs_dict['current']['yourIndex']
            
            # PPO Agent Turn
            if your_index == 0:
                obs = to_observation_class(obs_dict)
                sv_enc = get_encoder_input(obs, env.deck)
                legal_actions = [[i] for i in range(len(obs.select.option))]
                sv_dec = get_decoder_input(obs, legal_actions)
                
                enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32, device=device)
                enc_val = torch.tensor(sv_enc.value, dtype=torch.float32, device=device)
                enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32, device=device)
                dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32, device=device)
                dec_val = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
                dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
                
                with torch.no_grad():
                    val, logits = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
                    dist = torch.distributions.Categorical(logits=logits)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                
                chosen_action = legal_actions[action_idx.item()]
                
                states_enc.append((enc_idx, enc_val, enc_off))
                states_dec.append((dec_idx, dec_val, dec_off))
                actions.append(action_idx.item())
                log_probs_old.append(log_prob.item())
                values.append(val.item())
                
                obs_dict, reward, done = env.step(chosen_action)
                
                if done:
                    # Propagate final reward
                    r_arr = [0.0] * len(actions)
                    r_arr[-1] = reward
                    rewards = r_arr
                    break
                    
            # Heuristic Bot Turn (Opponent)
            else:
                try:
                    chosen_action = rule_agent(obs_dict)
                except Exception:
                    # Bot crashed, end episode as a loss for the bot (win for us)
                    r_arr = [0.0] * len(actions)
                    if len(r_arr) > 0: r_arr[-1] = 1.0
                    rewards = r_arr
                    break
                    
                obs_dict, reward, done = env.step(chosen_action)
                if done:
                    # Invert reward since opponent triggered it? No, tcg_env already normalizes:
                    # 0 = P0 win (1.0), 1 = P1 win (-1.0)
                    r_arr = [0.0] * len(actions)
                    if len(r_arr) > 0: r_arr[-1] = reward
                    rewards = r_arr
                    break
                    
            step_count += 1
            if step_count > 500: # safeguard
                r_arr = [0.0] * len(actions)
                rewards = r_arr
                break
                
        if len(rewards) == 0:
            print("Episode crashed immediately.")
            continue
            
        print(f"Collected Trajectory. Steps: {len(actions)}, Final Reward: {rewards[-1]}")
        
        advantages, returns = compute_gae(rewards, values)
        
        # PPO Update (batch_size=1 due to variable tensors, iterative optimization)
        for ppo_epoch in range(epochs_per_update):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            
            for i in range(len(actions)):
                enc_idx, enc_val, enc_off = states_enc[i]
                dec_idx, dec_val, dec_off = states_dec[i]
                old_log_prob = log_probs_old[i]
                adv = advantages[i]
                ret = returns[i]
                act = actions[i]
                
                val, logits = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(torch.tensor(act, device=device))
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2)
                
                value_loss = 0.5 * (val.squeeze() - ret)**2
                
                loss = policy_loss + value_loss - 0.01 * dist.entropy()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                
            print(f"PPO Update {ppo_epoch+1}/{epochs_per_update} - Ploss: {epoch_policy_loss/len(actions):.4f}, Vloss: {epoch_value_loss/len(actions):.4f}")

if __name__ == "__main__":
    run_ppo()
