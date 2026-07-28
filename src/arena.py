import time
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
            
    print(f"\nWin Rate vs Random: {wins_random}/100")
    print(f"Win Rate vs Greedy Bot: {wins_greedy}/100")
    
    with open("arena_results.txt", "w", encoding='utf-8') as f:
        f.write(f"{wins_random},{wins_greedy}")

if __name__ == "__main__":
    run_arena()
