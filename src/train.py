import os
import gc
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from env import PTCGEnv
from model import PokemonActorCritic
from ppo_buffer import PPOBuffer
from registry import register_model
from greedy_agent import GreedyAgent

def run_league_training():
    env = PTCGEnv()
    
    model = PokemonActorCritic()
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "latest_model.pt")
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded latest weights for League Training.")
        except RuntimeError:
            print("Failed to load latest_model.pt (Architecture mismatch). Using fresh weights.")
        
    # Past Agent
    past_model = PokemonActorCritic()
    past_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TITAN_LEAGUE_01.pt")
    if os.path.exists(past_path):
        try:
            past_checkpoint = torch.load(past_path, map_location='cpu')
            past_model.load_state_dict(past_checkpoint['model_state_dict'])
            print("Loaded past weights for League opponent.")
        except RuntimeError:
            print("Failed to load past_model (Architecture mismatch). Using fresh weights for past opponent.")
    past_model.eval()
    
    greedy_agent = GreedyAgent()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    buffer = PPOBuffer()
    
    num_episodes = 3000
    clip_param = 0.2
    entropy_coef = 0.05
    value_coef = 0.5
    ppo_epochs = 4
    
    # We update the scheduler every PPO epoch per episode.
    # We don't know exactly how many times buffer will have rewards, but max is num_episodes.
    # Let's just use a simple StepLR or manual warmup for PPO to be safer, because OneCycleLR is tricky with dynamic steps.
    # Actually, I'll just initialize scheduler as None here. Warmup was heavily requested, so let's use OneCycleLR.
    total_steps = num_episodes * ppo_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=0.1)
    
    start_time = time.time()
    last_value_loss = 0.0
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        
        # Select opponent for this game
        opponent_types = ["self", "random", "greedy", "past"]
        opponent_type = random.choice(opponent_types)
        
        # Randomly assign whether our PPO model is Player 0 or Player 1
        model_player_idx = random.choice([0, 1])
        opponent_player_idx = 1 - model_player_idx
        
        while not done and step < 200:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            current_player = int(state_vec[2])
            
            s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
            
            if current_player == model_player_idx:
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
            else:
                # Opponent acts
                valid_actions = np.where(mask == 1)[0]
                if len(valid_actions) == 0:
                    break
                    
                if opponent_type == "random":
                    action = int(np.random.choice(valid_actions))
                elif opponent_type == "greedy":
                    action = greedy_agent.act(obs)
                elif opponent_type == "past":
                    with torch.no_grad():
                        past_policy, _ = past_model(s_tensor, m_tensor)
                        past_p = past_policy.squeeze(0)[valid_actions]
                        if past_p.sum() > 0:
                            past_p /= past_p.sum()
                            action = int(np.random.choice(valid_actions, p=past_p.numpy()))
                        else:
                            action = int(np.random.choice(valid_actions))
                elif opponent_type == "self":
                    with torch.no_grad():
                        policy, _ = model(s_tensor, m_tensor)
                        p = policy.squeeze(0)[valid_actions]
                        if p.sum() > 0:
                            p /= p.sum()
                            action = int(np.random.choice(valid_actions, p=p.numpy()))
                        else:
                            action = int(np.random.choice(valid_actions))
                            
            try:
                next_obs, reward, is_done, _, _ = env.step(action)
            except:
                break
                
            done = is_done or env.is_done
            
            if current_player == model_player_idx:
                # If reward is relative to acting player, we store it.
                # env.py step() returns reward relative to the acting player.
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
                if scheduler is not None:
                    scheduler.step()
                last_value_loss = critic_loss.item()
                
            buffer.clear()
            
        if episode % 50 == 0:
            print(f"Episode {episode} | Value Loss: {last_value_loss:.4f} | Opponent: {opponent_type}", flush=True)

    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
    
    # Save a named checkpoint for registry
    named_ckpt = f"TITAN_TRANSFORMER_01.pt"
    named_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", named_ckpt)
    torch.save({'model_state_dict': model.state_dict()}, named_path)
    
    # Register model with placeholder values, to be updated by eval
    register_model("TITAN_TRANSFORMER_01", named_path, 0.0, 0.0)
    
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "value_loss.txt"), "w") as f:
        f.write(str(last_value_loss))
        
    print(f"League Training Complete. Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_league_training()
