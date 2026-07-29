import os
import sys
import gc
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.league_env import LeagueEnv
from src.model import PokemonActorCritic
from src.ppo_buffer import PPOBuffer

def run_league_ppo():
    print("Initializing League Environment...")
    env = LeagueEnv(ppo_ckpt_path="checkpoints/TOP_ELO_PPO_PEAK.pt")
    
    print("Initializing Active Model and KL Anchor...")
    model = PokemonActorCritic(num_layers=2)
    bc_model = PokemonActorCritic(num_layers=3)
    
    peak_path = os.path.join(base_dir, "checkpoints", "TOP_ELO_PPO_PEAK.pt")
    bc_path = os.path.join(base_dir, "checkpoints", "TOP_ELO_BC_MODEL_FINAL.pt")
    
    if os.path.exists(peak_path):
        checkpoint = torch.load(peak_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded Active Model from {peak_path}")
    else:
        print("ERROR: Peak model not found.")
        return
        
    if os.path.exists(bc_path):
        bc_checkpoint = torch.load(bc_path, map_location='cpu', weights_only=True)
        bc_model.load_state_dict(bc_checkpoint.get('model_state_dict', bc_checkpoint))
        print(f"Loaded BC Anchor Model from {bc_path}")
    else:
        print("ERROR: BC model not found.")
        return
        
    bc_model.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    buffer = PPOBuffer()
    
    num_episodes = 1000
    clip_param = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    kl_coef = 0.05
    ppo_epochs = 4
    
    wins = 0
    recent_wins = 0
    
    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 200:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            
            s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
            
            with torch.no_grad():
                policy, value = model(s_tensor, m_tensor)
                
            p = policy.squeeze(0)
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) > 0:
                p_valid = p[valid_actions]
                p_sum = p_valid.sum()
                if p_sum > 0:
                    p_valid /= p_sum
                    dist = torch.distributions.Categorical(p_valid)
                    action_idx = dist.sample()
                    action = int(valid_actions[action_idx.item()])
                    log_prob = dist.log_prob(action_idx).item()
                else:
                    action = int(np.random.choice(valid_actions))
                    log_prob = -np.log(len(valid_actions))
            else:
                break
                
            value_val = value.item()
            
            try:
                next_obs, reward, is_done, _, _ = env.step(action)
            except Exception as e:
                break
                
            done = is_done or env.is_done
            buffer.store(state_vec, mask, action, reward, value_val, log_prob, 1 - int(done))
            
            obs = next_obs
            step += 1
            
        if env.winner == env.agent_player_idx:
            wins += 1
            recent_wins += 1
            
        last_actor_loss = 0.0
        last_critic_loss = 0.0
        last_entropy = 0.0
        last_kl = 0.0
        
        if len(buffer.rewards) > 0:
            advantages, returns = buffer.compute_gae(0.0)
            
            b_states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
            b_masks = torch.tensor(np.array(buffer.action_masks), dtype=torch.float32)
            b_actions = torch.tensor(buffer.actions, dtype=torch.long)
            b_old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
            b_returns = torch.tensor(returns, dtype=torch.float32)
            b_advantages = torch.tensor(advantages, dtype=torch.float32)
            
            if len(b_advantages) > 1:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                
            # Precompute BC logits for KL
            with torch.no_grad():
                bc_policy, _ = bc_model(b_states, b_masks)
                bc_probs = torch.clamp(bc_policy, 1e-8, 1.0) * b_masks
                bc_probs = bc_probs / (bc_probs.sum(dim=1, keepdim=True) + 1e-8)
            
            for _ in range(ppo_epochs):
                policy, values = model(b_states, b_masks)
                values = values.squeeze(1)
                
                clamped_policy = torch.clamp(policy, 1e-8, 1.0)
                valid_probs = clamped_policy * b_masks
                valid_probs = valid_probs / (valid_probs.sum(dim=1, keepdim=True) + 1e-8)
                log_probs_valid = torch.log(valid_probs + 1e-8)
                entropy = -torch.sum(valid_probs * log_probs_valid * b_masks, dim=1).mean()
                
                # KL Divergence from BC
                bc_log_probs = torch.log(bc_probs + 1e-8)
                kl_div = torch.sum(bc_probs * (bc_log_probs - log_probs_valid) * b_masks, dim=1).mean()
                
                dist = torch.distributions.Categorical(clamped_policy)
                new_log_probs = dist.log_prob(b_actions)
                
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, b_returns)
                
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy + kl_coef * kl_div
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()
            last_entropy = entropy.item()
            last_kl = kl_div.item()
            buffer.clear()
            
        if episode % 100 == 0:
            win_rate = recent_wins / 100.0
            print(f"Episode {episode:04d} | WR: {win_rate:.2%} | Opponent: {env.current_opponent_name:<20} | Entropy: {last_entropy:.4f} | KL: {last_kl:.4f}")
            recent_wins = 0
            
            # Aggressive checkpointing
            ckpt_save_path = os.path.join(base_dir, "checkpoints", f"TITAN_LEAGUE_PPO_ep{episode}.pt")
            torch.save({'model_state_dict': model.state_dict()}, ckpt_save_path)
            
        elif episode > 995:
            # Print final 5 episodes strictly
            print(f"Episode {episode:04d} | Opponent: {env.current_opponent_name:<20} | Entropy: {last_entropy:.4f} | KL: {last_kl:.4f} | Actor: {last_actor_loss:.4f} | Critic: {last_critic_loss:.4f}")

    print(f"\nLeague PPO Training Complete. Overall WR: {wins/num_episodes:.2%}. Time: {time.time()-start_time:.1f}s")
    
if __name__ == "__main__":
    run_league_ppo()
