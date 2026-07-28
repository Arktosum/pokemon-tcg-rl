import time
import torch
import torch.optim as optim
from env import PTCGEnv
from model import PokemonAlphaNet
from replay_buffer import ReplayBuffer
from train import SelfPlayTrainer
import os

def run_rl():
    print("--- Phase 8: AlphaZero Self-Play RL ---")
    env = PTCGEnv()
    model = PokemonAlphaNet()
    if os.path.exists("checkpoints/latest_model.pt"):
        checkpoint = torch.load("checkpoints/latest_model.pt", map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded BC Epoch 20 weights.")
        
    buffer = ReplayBuffer(capacity=10000)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    trainer = SelfPlayTrainer(model, optimizer, buffer)
    
    num_games = 1000
    start_time = time.perf_counter()
    
    last_p_loss = 0.0
    last_v_loss = 0.0
    
    for game in range(1, num_games + 1):
        steps = trainer.generate_episode(env, num_simulations=50) # 10 to keep it fast
        p_loss, v_loss = trainer.train_step(batch_size=min(32, len(buffer)))
        
        if p_loss is not None:
            last_p_loss = p_loss
            last_v_loss = v_loss
            
        if game % 25 == 0:
            print(f"Game {game}/{num_games} | Buffer: {len(buffer)} | P-Loss: {last_p_loss:.4f} | V-Loss: {last_v_loss:.4f}")
            
    trainer.save_checkpoint("checkpoints/latest_model.pt")
    
    with open("rl_results.txt", "w", encoding='utf-8') as f:
        f.write(f"{last_p_loss:.4f},{last_v_loss:.4f}")
        
    print(f"RL Training Complete. Time: {time.perf_counter()-start_time:.2f}s")

if __name__ == "__main__":
    run_rl()
