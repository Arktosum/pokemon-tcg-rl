import os

# 1. puct.py (Change pb_c_init and num_simulations)
with open("puct.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("pb_c_init = 1.25", "pb_c_init = 1.5")
content = content.replace("num_simulations=10", "num_simulations=50")
with open("puct.py", "w", encoding="utf-8") as f:
    f.write(content)

# 2. bc_train.py (Change epochs to 30)
with open("bc_train.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("epochs = 20", "epochs = 30")
content = content.replace("Epoch 20 BC Policy Loss", "Epoch 30 BC Policy Loss")
with open("bc_train.py", "w", encoding="utf-8") as f:
    f.write(content)

# 3. rl_train.py (Change num_games to 1000, num_simulations to 50)
with open("rl_train.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("num_games = 250", "num_games = 1000")
content = content.replace("num_simulations=10", "num_simulations=50")
with open("rl_train.py", "w", encoding="utf-8") as f:
    f.write(content)

# 4. Wipe poisoned weights
if os.path.exists("checkpoints/latest_model.pt"):
    os.remove("checkpoints/latest_model.pt")
if os.path.exists("best_model.pt"):
    os.remove("best_model.pt")

# 5. replay_parser.py and arena.py
with open("replay_parser.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("num_simulations=10", "num_simulations=50")
with open("replay_parser.py", "w", encoding="utf-8") as f:
    f.write(content)

with open("arena.py", "r", encoding="utf-8") as f:
    content = f.read()
content = content.replace("num_simulations=10", "num_simulations=50")
with open("arena.py", "w", encoding="utf-8") as f:
    f.write(content)

# 6. Write eval_strict.py
eval_code = """import time
import numpy as np
import torch
from env import PTCGEnv
from model import PokemonAlphaNet
from puct import PUCTSearch
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
            searcher = PUCTSearch(model, num_simulations=50)
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
    model = PokemonAlphaNet()
    if os.path.exists("checkpoints/latest_model.pt"):
        checkpoint = torch.load("checkpoints/latest_model.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    wins = 0
    for i in range(100):
        wins += play_match(env, model, "random")
        if (i+1)%10 == 0:
            print(f"Eval: {wins}/{i+1}")
        
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

print("Phase 11 Setup Complete.")
