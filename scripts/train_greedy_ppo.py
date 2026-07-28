import os
import gc
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from env import PTCGEnv
from model import PokemonActorCritic
from ppo_buffer import PPOBuffer
from greedy_agent import GreedyAgent

def run_greedy_ppo():
    env = PTCGEnv()
    model = PokemonActorCritic()
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TITAN_TRANSFORMER_LEAGUE_01.pt")
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TITAN_GREEDY_PPO_01.pt")
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded starting weights from {ckpt_path}")
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    buffer = PPOBuffer()
    greedy_agent = GreedyAgent()
    
    num_episodes = 250
    clip_param = 0.2
    entropy_coef = 0.05
    value_coef = 0.5
    ppo_epochs = 4
    
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_entropy = 0.0
    first_entropy = None
    
    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 200:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            current_player = int(state_vec[2])
            
            if current_player == 1:
                action = greedy_agent.act(obs)
                # Greedy agent doesn't need RL states tracked
                value_val = 0.0
                log_prob = 0.0
            else:
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
            except:
                # If engine crashes, just end the episode
                break
                
            done = is_done or env.is_done
            
            # Store only Model (player 0) experiences
            if current_player == 0:
                buffer.store(state_vec, mask, action, reward, value_val, log_prob, 1 - int(done))
            
            obs = next_obs
            step += 1
            
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
            
            for _ in range(ppo_epochs):
                policy, values = model(b_states, b_masks)
                values = values.squeeze(1)
                
                clamped_policy = torch.clamp(policy, 1e-8, 1.0)
                
                valid_probs = clamped_policy * b_masks
                valid_probs = valid_probs / (valid_probs.sum(dim=1, keepdim=True) + 1e-8)
                log_probs_valid = torch.log(valid_probs + 1e-8)
                entropy = -torch.sum(valid_probs * log_probs_valid * b_masks, dim=1).mean()
                
                dist = torch.distributions.Categorical(clamped_policy)
                new_log_probs = dist.log_prob(b_actions)
                
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, b_returns)
                
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()
            last_entropy = entropy.item()
            if first_entropy is None:
                first_entropy = last_entropy
            buffer.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        if episode % 10 == 0:
            print(f"Episode {episode} | Mean Policy Entropy: {last_entropy:.4f} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f}", flush=True)
            
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"PPO Training Complete. Saved to {save_path}. Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_greedy_ppo()
