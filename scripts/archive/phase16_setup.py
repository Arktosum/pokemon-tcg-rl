import os
import re

# Update 03_META_RESEARCH.md
with open("03_META_RESEARCH.md", "a", encoding="utf-8") as f:
    f.write("\n## Reward Shaping (Phase 16)\n")
    f.write("- **Win:** +1.0\n")
    f.write("- **Loss:** -1.0\n")
    f.write("- **Dense Rewards (Irreversible Progression):** Taking a Prize Card = +0.1, Opponent taking Prize Card = -0.1.\n")

# Update 02_EXPERIMENT_TRACKER.md
with open("02_EXPERIMENT_TRACKER.md", "a", encoding="utf-8") as f:
    f.write("\n| `016` | 2026-07-28 13:46 | Phase 16: Dense Reward Shaping | N/A | Added +/- 0.1 per prize card taken, 2500 episodes | [ACTIVE LOCK] |\n")

# Update env.py
with open("src/env.py", "r", encoding="utf-8") as f:
    env_content = f.read()

# We need to replace the step function
new_step = """    def step(self, action):
        if self.is_done:
            return {"obs": np.zeros(120, dtype=np.float32), "action_mask": np.zeros(MAX_OPTIONS, dtype=np.int8)}, 0.0, True, False, {}
            
        obs = self.current_obs
        select_list = []
        acting_player = obs.current.yourIndex
        
        # Save prize counts before stepping
        p0_prizes_before = len(obs.current.players[0].prize)
        p1_prizes_before = len(obs.current.players[1].prize)
        
        if obs.select is not None:
            opts = obs.select.option
            num_opts = len(opts)
            
            if action < num_opts:
                select_list.append(int(action))
            else:
                select_list.append(random.randint(0, num_opts - 1))
                
            min_c = obs.select.minCount
            available = list(range(num_opts))
            if select_list[0] in available:
                available.remove(select_list[0])
                
            while len(select_list) < min_c and available:
                nxt = random.choice(available)
                select_list.append(nxt)
                available.remove(nxt)
                
        try:
            obs_dict = battle_select(select_list)
            next_state = self._process_obs(obs_dict)
        except Exception as e:
            self.is_done = True
            return {"obs": np.zeros(120, dtype=np.float32), "action_mask": np.zeros(MAX_OPTIONS, dtype=np.int8)}, -1.0, True, False, {}

        reward = 0.0
        if self.is_done:
            if self.winner == acting_player:
                reward = 1.0
            elif self.winner == 1 - acting_player:
                reward = -1.0
            else:
                reward = 0.0
        else:
            p0_prizes_after = len(self.current_obs.current.players[0].prize)
            p1_prizes_after = len(self.current_obs.current.players[1].prize)
            p0_taken = p0_prizes_before - p0_prizes_after
            p1_taken = p1_prizes_before - p1_prizes_after
            
            if acting_player == 0:
                reward += (p0_taken * 0.1) - (p1_taken * 0.1)
            else:
                reward += (p1_taken * 0.1) - (p0_taken * 0.1)
                
        return next_state, reward, self.is_done, False, {}"""

env_content = re.sub(r'    def step\(self, action\):.*?(?=    def close\(self\):)', new_step + "\n\n", env_content, flags=re.DOTALL)

with open("src/env.py", "w", encoding="utf-8") as f:
    f.write(env_content)

# Update eval_strict.py to run both random and greedy
eval_code = """import time
import numpy as np
import torch
from env import PTCGEnv
from model import PokemonActorCritic
import os
from greedy_agent import GreedyAgent

def play_match(env, model, opponent_type):
    obs, _ = env.reset()
    done = False
    step = 0
    greedy_agent = GreedyAgent()
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
            elif opponent_type == "greedy":
                action = greedy_agent.act(obs)
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
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "latest_model.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    wins_random = 0
    for i in range(100):
        wins_random += play_match(env, model, "random")
        
    wins_greedy = 0
    for i in range(100):
        wins_greedy += play_match(env, model, "greedy")
        
    print(f"Win Rate vs Random: {wins_random}/100")
    print(f"Win Rate vs Repaired Greedy: {wins_greedy}/100")
    if wins_random < 95 or wins_greedy < 95:
        print("FAILED CRITERIA")
    else:
        print("PASSED CRITERIA")

if __name__ == "__main__":
    run_eval()
"""
with open("src/eval_strict.py", "w", encoding="utf-8") as f:
    f.write(eval_code)

# Update ppo_train.py to 2500 episodes
with open("src/ppo_train.py", "r", encoding="utf-8") as f:
    ppo_content = f.read()
ppo_content = ppo_content.replace("num_episodes = 2000", "num_episodes = 2500")
with open("src/ppo_train.py", "w", encoding="utf-8") as f:
    f.write(ppo_content)

# Purge model weights
ckpt_path = os.path.join("checkpoints", "latest_model.pt")
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

print("Setup Complete Phase 16.")
