import os
import sys
import glob
import time
import torch
import torch.nn.functional as F
import concurrent.futures
import multiprocessing

# Add necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02_ppo_training')))

from env_wrapper import PokemonEnvWrapper
from model import TitanTransformer

# Import opponents
from greedy_agent import greedy_agent
from tactical_agent import tactical_agent
from aggro_agent import aggro_agent
from setup_agent import setup_agent

def random_agent(obs_dict):
    import random
    import os
    from cg.api import to_observation_class
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent', 'deck.csv'))
        if not os.path.exists(file_path):
            file_path = "/kaggle_simulations/agent/deck.csv"
        try:
            with open(file_path, "r") as file:
                return [int(c) for c in file.read().split("\n") if c.strip()]
        except:
            return [1] * 60
    
    options = obs.select.option
    max_count = obs.select.maxCount
    if max_count == 0 or not options: return []
    try:
        count = max_count if len(options) >= max_count else len(options)
        return random.sample(list(range(len(options))), count)
    except:
        return [0] * min(max_count, len(options))

OPPONENTS = {
    "RandomBot": random_agent,
    "SetupBot": setup_agent,
    "AggroBot": aggro_agent,
    "GreedyBot": greedy_agent,
    "TacticalBot": tactical_agent
}

def find_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(os.path.dirname(__file__), "..", "02_ppo_training", "*_titan_bc.pt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]

def select_action(logits):
    action = torch.argmax(logits, dim=-1)
    return action.item()

def run_match(opponent_name, match_id):
    # Limit threads to prevent overloading CPU
    torch.set_num_threads(1)
    
    device = torch.device("cpu")
    model = TitanTransformer().to(device)
    
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    env = PokemonEnvWrapper(OPPONENTS[opponent_name])
    obs = env.reset()
    
    steps = 0
    done = False
    
    while not done:
        enc_i = obs["enc_indices"].to(device)
        enc_o = obs["enc_offsets"].to(device)
        enc_w = obs["enc_weights"].to(device)
        dec_i = obs["dec_indices"].to(device)
        dec_o = obs["dec_offsets"].to(device)
        dec_w = obs["dec_weights"].to(device)
        
        with torch.no_grad():
            logits, _ = model(enc_i, enc_o, enc_w, dec_i, dec_o, dec_w)
            action = select_action(logits)
            
        obs, reward, done, info = env.step(action)
        steps += 1
        
        if done:
            engine_reward = info.get("engine_reward", 0)
            if engine_reward == 1:
                result = "Win"
            elif engine_reward == -1:
                result = "Loss"
            else:
                result = "Draw"
            return opponent_name, result, steps
            
    return opponent_name, "Draw", steps

def main():
    num_matches = 100
    
    results = {
        name: {"Win": 0, "Loss": 0, "Draw": 0, "TotalSteps": 0} for name in OPPONENTS
    }
    
    tasks = []
    for name in OPPONENTS:
        for i in range(num_matches):
            tasks.append((name, i))
            
    completed = 0
    total = len(tasks)
    start_time = time.time()
    
    print(f"Starting {num_matches} matches against {len(OPPONENTS)} opponents (Total: {total})...")
    
    # ProcessPoolExecutor with 8 workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_match, *task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                opp_name, res, steps = future.result()
                results[opp_name][res] += 1
                results[opp_name]["TotalSteps"] += steps
                completed += 1
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{completed}/{total}] Matches finished. Elapsed: {elapsed:.1f}s")
            except Exception as e:
                print(f"Match failed: {e}")
                
    print("\n\n# Evaluation Matrix\n")
    print("| Opponent | BC Win Rate | Opp. Win Rate | Draws | Avg Game Length |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for name in OPPONENTS:
        stats = results[name]
        total_games = stats["Win"] + stats["Loss"] + stats["Draw"]
        if total_games == 0: continue
        
        bc_win_rate = (stats["Win"] / total_games) * 100
        opp_win_rate = (stats["Loss"] / total_games) * 100
        draw_rate = (stats["Draw"] / total_games) * 100
        avg_steps = stats["TotalSteps"] / total_games
        
        print(f"| `{name}` | {bc_win_rate:.1f}% | {opp_win_rate:.1f}% | {draw_rate:.1f}% | {avg_steps:.1f} |")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
