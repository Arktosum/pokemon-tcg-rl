import time
import numpy as np
import torch
from env import PTCGEnv
from model import PokemonAlphaNet
from puct import PUCTSearch

def run_evaluation():
    env = PTCGEnv()
    model = PokemonAlphaNet()
    try:
        checkpoint = torch.load("checkpoints/latest_model.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")
    model.eval()
    
    wins = 0
    total_games = 10
    total_time = 0
    moves = 0
    
    for game in range(total_games):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            current_player = int(state_vec[2])
            
            start_t = time.perf_counter()
            if current_player == 0:
                # AlphaNet
                searcher = PUCTSearch(model, num_simulations=10) # 10 for speed
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
                total_time += (time.perf_counter() - start_t)
                moves += 1
            else:
                # Random Agent
                valid_actions = np.where(mask == 1)[0]
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                else:
                    break
                    
            obs, reward, is_done, _, _ = env.step(action)
            done = is_done or env.is_done
            step += 1
            
        winner = env.winner if hasattr(env, 'winner') else -1
        if winner == 0:
            wins += 1
        print(f"Game {game+1} finished. Winner: {winner}")
        
    avg_time = total_time / max(1, moves)
    print(f"Win Rate vs Random: {wins}/{total_games}")
    print(f"Average PUCT Time per move: {avg_time:.4f}s")
    
    with open("eval_results.txt", "w") as f:
        f.write(f"{wins},{avg_time:.4f}")

if __name__ == "__main__":
    run_evaluation()
