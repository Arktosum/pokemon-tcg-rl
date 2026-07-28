import os

# 1. Update Tracker and Meta Research
with open("02_EXPERIMENT_TRACKER.md", "a", encoding="utf-8") as f:
    f.write("\n| `013` | 2026-07-28 13:12 | Phase 13: Architecture Pivot to PPO | N/A | Abandoned MCTS, Built Actor-Critic PPO | [ACTIVE LOCK] |\n")

with open("03_META_RESEARCH.md", "a", encoding="utf-8") as f:
    f.write("\n\n## Phase 13: Proximal Policy Optimization (PPO)\n\nDue to Kaggle C++ engine state cloning limitations, MCTS is structurally impossible. We pivot to PPO. The network is renamed PokemonActorCritic. Rollouts are stored in PPOBuffer, Generalized Advantage Estimation (GAE) is applied, and weights are updated via PPO Clipped Objective.\n\n")

# 2. Wipe poisoned weights and puct
if os.path.exists("puct.py"): os.remove("puct.py")
if os.path.exists("checkpoints/latest_model.pt"): os.remove("checkpoints/latest_model.pt")
if os.path.exists("best_model.pt"): os.remove("best_model.pt")

# 3. Rename model class
with open("model.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("class PokemonAlphaNet", "class PokemonActorCritic")
with open("model.py", "w", encoding="utf-8") as f:
    f.write(content)
    
with open("bc_train.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("PokemonAlphaNet", "PokemonActorCritic")
with open("bc_train.py", "w", encoding="utf-8") as f:
    f.write(content)

# 4. Write ppo_buffer.py
ppo_buffer_code = """import numpy as np
import torch

class PPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.action_masks = []
        self.gamma = gamma
        self.lam = lam
        
    def store(self, state, action_mask, action, reward, value, log_prob, mask):
        self.states.append(state)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.action_masks.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()
        
    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae = 0
        returns = []
        advantages = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + self.gamma * self.lam * self.masks[step] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return advantages, returns
"""
with open("ppo_buffer.py", "w", encoding="utf-8") as f:
    f.write(ppo_buffer_code)

# 5. Write ppo_train.py
ppo_train_code = """import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from env import PTCGEnv
from model import PokemonActorCritic
from ppo_buffer import PPOBuffer

def run_ppo():
    env = PTCGEnv()
    model = PokemonActorCritic()
    if os.path.exists("checkpoints/latest_model.pt"):
        checkpoint = torch.load("checkpoints/latest_model.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded BC weights for PPO.")
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    buffer = PPOBuffer()
    
    num_episodes = 500
    clip_param = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    ppo_epochs = 4
    
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    
    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 200:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            current_player = int(state_vec[2])
            
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
                    action = valid_actions[action_idx.item()]
                    log_prob = dist.log_prob(action_idx).item()
                else:
                    action = int(np.random.choice(valid_actions))
                    log_prob = -np.log(len(valid_actions))
            else:
                break
                
            try:
                next_obs, reward, is_done, _, _ = env.step(action)
            except:
                break
                
            done = is_done or env.is_done
            buffer.store(state_vec, mask, action, reward, value.item(), log_prob, 1 - int(done))
            
            obs = next_obs
            step += 1
            
        if len(buffer.rewards) > 0:
            advantages, returns = buffer.compute_gae(0.0)
            
            b_states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
            b_masks = torch.tensor(np.array(buffer.action_masks), dtype=torch.int8)
            b_actions = torch.tensor(buffer.actions, dtype=torch.long)
            b_old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
            b_returns = torch.tensor(returns, dtype=torch.float32)
            b_advantages = torch.tensor(advantages, dtype=torch.float32)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            for _ in range(ppo_epochs):
                policy, values = model(b_states, b_masks)
                values = values.squeeze(1)
                
                clamped_policy = torch.clamp(policy, 1e-8, 1.0)
                dist = torch.distributions.Categorical(clamped_policy)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
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
            buffer.clear()
            
        if episode % 50 == 0:
            print(f"Episode {episode} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f}")
            
    torch.save({'model_state_dict': model.state_dict()}, "checkpoints/latest_model.pt")
    with open("ppo_results.txt", "w") as f:
        f.write(f"{last_actor_loss:.4f},{last_critic_loss:.4f}")
    print(f"PPO Training Complete. Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    run_ppo()
"""
with open("ppo_train.py", "w", encoding="utf-8") as f:
    f.write(ppo_train_code)

# 6. Write eval_strict.py for PPO
eval_code = """import time
import numpy as np
import torch
from env import PTCGEnv
from model import PokemonActorCritic
import os

def play_match(env, model, opponent_type):
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
            if opponent_type == "random":
                valid_actions = np.where(mask == 1)[0]
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                else:
                    break
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
    if os.path.exists("checkpoints/latest_model.pt"):
        checkpoint = torch.load("checkpoints/latest_model.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    wins = 0
    start_time = time.time()
    for i in range(100):
        wins += play_match(env, model, "random")
        
    print(f"Win Rate vs Random: {wins}/100")
    if wins < 95:
        print("FAILED CRITERIA")
    else:
        print("PASSED CRITERIA")

if __name__ == "__main__":
    run_eval()
"""
with open("eval_strict.py", "w", encoding="utf-8") as f:
    f.write(eval_code)

print("Phase 13 Setup Complete.")
