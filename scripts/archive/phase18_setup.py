import os
import re

# Update 03_META_RESEARCH.md
meta_append = r"""
### Phase 18 Curriculum Rewards
- **Dense Progression Rewards:** Add +0.05 for Bench-filling (max 5) and +0.05 for Energy Attachment (max 3 per mon).
- **The Safety Lock:** Implemented via max total bounds per episode to prevent infinite loop reward farming.
- **Robustness Training (Domain Randomization):** 30% of self-play games are against a Random Agent to force OOD robustness.
"""
with open("03_META_RESEARCH.md", "a", encoding="utf-8") as f:
    f.write(meta_append)

# Update 02_EXPERIMENT_TRACKER.md
exp_append = r"""
| 018 | Phase 18: Curriculum Learning & Noise Training | Curriculum Rewards (+0.05 bench/energy) + 30% Random opponent | [ACTIVE LOCK] |
"""
with open("02_EXPERIMENT_TRACKER.md", "a", encoding="utf-8") as f:
    f.write(exp_append)

# Update env.py
env_path = os.path.join("src", "env.py")
with open(env_path, "r") as f:
    env_content = f.read()

env_content = env_content.replace(
    "self.deck1 = self._read_deck()\n        \n        obs_dict",
    "self.deck1 = self._read_deck()\n        \n        self.max_bench_size = [0, 0]\n        self.max_total_energy = [0, 0]\n        \n        obs_dict"
)

old_reward_logic = """        else:
            p0_prizes_after = len(self.current_obs.current.players[0].prize)
            p1_prizes_after = len(self.current_obs.current.players[1].prize)
            p0_taken = p0_prizes_before - p0_prizes_after
            p1_taken = p1_prizes_before - p1_prizes_after
            
            if acting_player == 0:
                reward += (p0_taken * 0.1) - (p1_taken * 0.1)
            else:
                reward += (p1_taken * 0.1) - (p0_taken * 0.1)"""

new_reward_logic = """        else:
            p0_prizes_after = len(self.current_obs.current.players[0].prize)
            p1_prizes_after = len(self.current_obs.current.players[1].prize)
            p0_taken = p0_prizes_before - p0_prizes_after
            p1_taken = p1_prizes_before - p1_prizes_after
            
            p0_bench_after = len(self.current_obs.current.players[0].bench)
            p1_bench_after = len(self.current_obs.current.players[1].bench)
            
            p0_energy_after = 0
            if self.current_obs.current.players[0].active and self.current_obs.current.players[0].active[0]:
                p0_energy_after += len(self.current_obs.current.players[0].active[0].energies)
            for pk in self.current_obs.current.players[0].bench:
                p0_energy_after += len(pk.energies)
                
            p1_energy_after = 0
            if self.current_obs.current.players[1].active and self.current_obs.current.players[1].active[0]:
                p1_energy_after += len(self.current_obs.current.players[1].active[0].energies)
            for pk in self.current_obs.current.players[1].bench:
                p1_energy_after += len(pk.energies)
                
            curr_reward = 0.0
            if p0_bench_after > self.max_bench_size[0]:
                if acting_player == 0: curr_reward += (p0_bench_after - self.max_bench_size[0]) * 0.05
                self.max_bench_size[0] = p0_bench_after
                
            if p1_bench_after > self.max_bench_size[1]:
                if acting_player == 1: curr_reward += (p1_bench_after - self.max_bench_size[1]) * 0.05
                self.max_bench_size[1] = p1_bench_after
                
            if p0_energy_after > self.max_total_energy[0]:
                if acting_player == 0: curr_reward += (p0_energy_after - self.max_total_energy[0]) * 0.05
                self.max_total_energy[0] = p0_energy_after
                
            if p1_energy_after > self.max_total_energy[1]:
                if acting_player == 1: curr_reward += (p1_energy_after - self.max_total_energy[1]) * 0.05
                self.max_total_energy[1] = p1_energy_after
            
            if acting_player == 0:
                reward += (p0_taken * 0.1) - (p1_taken * 0.1) + curr_reward
            else:
                reward += (p1_taken * 0.1) - (p0_taken * 0.1) + curr_reward"""

env_content = env_content.replace(old_reward_logic, new_reward_logic)
with open(env_path, "w") as f:
    f.write(env_content)

# Update ppo_train.py
ppo_path = os.path.join("src", "ppo_train.py")
with open(ppo_path, "r") as f:
    ppo_content = f.read()

ppo_content = ppo_content.replace("import numpy as np", "import random\nimport numpy as np")
ppo_content = ppo_content.replace("num_episodes = 2000", "num_episodes = 3000")

old_loop = """    for episode in range(1, num_episodes + 1):
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
                    action = int(valid_actions[action_idx.item()])
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
            step += 1"""

new_loop = """    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        
        # 30% Domain Randomization
        is_random_opponent = (random.random() < 0.3)
        random_player_idx = 1 if is_random_opponent else -1
        
        while not done and step < 200:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            current_player = int(state_vec[2])
            
            if current_player == random_player_idx:
                valid_actions = np.where(mask == 1)[0]
                action = int(np.random.choice(valid_actions))
                log_prob = -np.log(len(valid_actions))
                value_val = 0.0
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
                break
                
            done = is_done or env.is_done
            
            # Store only non-random agent experiences
            if current_player != random_player_idx:
                buffer.store(state_vec, mask, action, reward, value_val, log_prob, 1 - int(done))
            
            obs = next_obs
            step += 1"""

ppo_content = ppo_content.replace(old_loop, new_loop)
with open(ppo_path, "w") as f:
    f.write(ppo_content)

print("Setup Complete Phase 18.")
