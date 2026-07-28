import os

def double_space_markdown(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    import re
    content = re.sub(r'\n+', '\n\n', content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip() + '\n\n')

# 1. Update 00_DIRECTIVES.md
with open("00_DIRECTIVES.md", "a", encoding='utf-8') as f:
    f.write("\n\nNote (Phase 7): Acknowledged the 5-submission-per-day Kaggle constraint and the necessity of strict local validation against Greedy Agents before submission.\n\n")

# 2. Update 05_RAW_RESEARCH_ARCHIVE.md
with open("05_RAW_RESEARCH_ARCHIVE.md", "a", encoding='utf-8') as f:
    f.write("\n\n### Web Search Results: PTCG Heuristic Greedy Agents (2026-07-28)\n\n*   **Performance Benchmark:** Used to establish a floor. A simple rule-based agent employs greedy logic - prioritizing moves that deal the most immediate damage, attaching energy to active Pokemon, or evolving whenever possible.\n\n*   **Implementation:** They evaluate based on immediate metrics. For our environment, we prioritize non-Pass actions (e.g., Action > 0), strictly filtering through `action_mask` to prevent infinite loops (trying to play cards we can't afford).\n\n")

# 3. Update 02_EXPERIMENT_TRACKER.md
with open("02_EXPERIMENT_TRACKER.md", "a", encoding='utf-8') as f:
    f.write("\n\n| `007` | 2026-07-28 12:38 | Phase 7: Scale-Up & Arena Eval | N/A | Massive BC and Greedy Agent Evaluation | [ACTIVE LOCK] |\n\n")

# 4. Fix Markdown
double_space_markdown("01_JOURNEY_LOG.md")
double_space_markdown("02_EXPERIMENT_TRACKER.md")
double_space_markdown("00_DIRECTIVES.md")
double_space_markdown("05_RAW_RESEARCH_ARCHIVE.md")

# 5. Greedy Agent
greedy_code = """import numpy as np

class GreedyAgent:
    def __init__(self):
        pass
        
    def act(self, obs):
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) == 0:
            return 0
            
        best_action = 0
        for a in reversed(valid_actions):
            if a != 0:
                best_action = int(a)
                break
                
        if best_action == 0 and len(valid_actions) > 0:
            best_action = int(valid_actions[0])
            
        return best_action
"""
with open("greedy_agent.py", "w", encoding='utf-8') as f:
    f.write(greedy_code)

# 6. Arena
arena_code = """import time
import numpy as np
import torch
from env import PTCGEnv
from model import PokemonAlphaNet
from puct import PUCTSearch
from greedy_agent import GreedyAgent

def play_match(env, model, opponent_type):
    obs, _ = env.reset()
    done = False
    step = 0
    greedy = GreedyAgent()
    
    while not done and step < 200:
        state_vec = obs["obs"]
        mask = obs["action_mask"]
        current_player = int(state_vec[2])
        
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
                break
        else:
            if opponent_type == "random":
                valid_actions = np.where(mask == 1)[0]
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                else:
                    break
            else:
                action = greedy.act(obs)
                
        obs, reward, is_done, _, _ = env.step(action)
        done = is_done or env.is_done
        step += 1
        
    winner = env.winner if hasattr(env, 'winner') else -1
    return 1 if winner == 0 else 0

def run_arena():
    env = PTCGEnv()
    model = PokemonAlphaNet()
    try:
        checkpoint = torch.load("checkpoints/latest_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
    except:
        pass
    model.eval()
    
    print("Arena: AlphaNet vs Random (100 games)")
    wins_random = 0
    for i in range(100):
        wins_random += play_match(env, model, "random")
        if (i+1) % 10 == 0:
            print(f"Played {i+1} random games...")
            
    print("Arena: AlphaNet vs Greedy (100 games)")
    wins_greedy = 0
    for i in range(100):
        wins_greedy += play_match(env, model, "greedy")
        if (i+1) % 10 == 0:
            print(f"Played {i+1} greedy games...")
            
    print(f"\\nWin Rate vs Random: {wins_random}/100")
    print(f"Win Rate vs Greedy Bot: {wins_greedy}/100")
    
    with open("arena_results.txt", "w", encoding='utf-8') as f:
        f.write(f"{wins_random},{wins_greedy}")

if __name__ == "__main__":
    run_arena()
"""
with open("arena.py", "w", encoding='utf-8') as f:
    f.write(arena_code)

# 7. Update replay_parser.py
with open("replay_parser.py", "r", encoding='utf-8') as f:
    content = f.read()
if "for _ in range(50): trainer.generate_episode" not in content:
    content = content.replace("trainer.generate_episode(env, num_simulations=10)", "for _ in range(50): trainer.generate_episode(env, num_simulations=10)")
    with open("replay_parser.py", "w", encoding='utf-8') as f:
        f.write(content)
# Remove the old dummy file to force regeneration
if os.path.exists("data/replays/episode_mock.json"):
    os.remove("data/replays/episode_mock.json")

# 8. Update bc_train.py
with open("bc_train.py", "r", encoding='utf-8') as f:
    content = f.read()
if "epochs = 20" not in content:
    content = content.replace("epochs = 5", "epochs = 20")
    content = content.replace("Epoch 5 BC Policy Loss", "Epoch 20 BC Policy Loss")
    with open("bc_train.py", "w", encoding='utf-8') as f:
        f.write(content)

print("Phase 7 Setup Complete.")
