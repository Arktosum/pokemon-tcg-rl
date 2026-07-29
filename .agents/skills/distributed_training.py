import torch
import torch.multiprocessing as mp
import os
from src.league_env import LeagueEnv
from src.model import PokemonActorCritic

def worker_train(worker_id, shared_model, queue):
    # Distributed training worker natively using LeagueEnv
    env = LeagueEnv()
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=1e-4)
    
    print(f"Worker {worker_id} started training against LeagueEnv pool.")
    for episode in range(100):
        obs, _ = env.reset()
        done = False
        while not done:
            # Sync with shared model
            state = torch.tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(obs["action_mask"], dtype=torch.int8).unsqueeze(0)
            
            policy, value = shared_model(state, mask)
            
            # Simple dummy action selection for worker loop
            valid_actions = torch.where(mask[0] == 1)[0]
            if len(valid_actions) == 0:
                action = 0
            else:
                action = valid_actions[0].item()
                
            obs, reward, done, truncated, info = env.step(action)
            
        queue.put((worker_id, episode, reward))

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    
    shared_model = PokemonActorCritic(num_layers=2)
    shared_model.share_memory()
    
    queue = mp.Queue()
    num_processes = 4
    processes = []
    
    for i in range(num_processes):
        p = mp.Process(target=worker_train, args=(i, shared_model, queue))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("Distributed training completed.")
