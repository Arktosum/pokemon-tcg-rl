import time
import math
import torch
import torch.optim as optim
from env import PTCGEnv
from model import PokemonAlphaNet
from replay_buffer import ReplayBuffer
from train import SelfPlayTrainer

def run_test():
    print("--- Phase 4: Self-Play Training Loop Validation ---")
    
    # 1. Init
    env = PTCGEnv()
    model = PokemonAlphaNet()
    # Weight decay c=1e-4 as per blueprint
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=1000)
    
    trainer = SelfPlayTrainer(model, optimizer, buffer)
    
    # 2. Generate Episodes until enough data
    print("Generating Self-Play Episodes (N=25) until buffer size >= 32...")
    start_time = time.perf_counter()
    total_steps = 0
    episodes = 0
    while len(buffer) < 32:
        steps = trainer.generate_episode(env, num_simulations=25)
        total_steps += steps
        episodes += 1
    ep_time = time.perf_counter() - start_time
    
    print(f"Generated {episodes} episodes ({total_steps} steps).")
    print(f"Time taken: {ep_time:.2f} seconds.")
    print(f"Buffer size: {len(buffer)}")
    
    # 3. Mini-batch Training Step
    print("\nExecuting 1 Training Step (Batch Size: 32 for test)...")
    p_loss, v_loss = trainer.train_step(batch_size=32)
    
    if p_loss is not None and v_loss is not None:
        print(f"Policy Loss: {p_loss:.4f}")
        print(f"Value Loss: {v_loss:.4f}")
        print(f"Total Loss: {p_loss + v_loss:.4f}")
        
        assert not math.isnan(p_loss) and not math.isinf(p_loss), "Policy Loss invalid (NaN/Inf)!"
        assert not math.isnan(v_loss) and not math.isinf(v_loss), "Value Loss invalid (NaN/Inf)!"
    else:
        print("Not enough data to train.")
        
    # 4. Checkpoint
    trainer.save_checkpoint("checkpoints/latest_model.pt")
    print("\nValidation Complete. Checkpoint saved.")

if __name__ == "__main__":
    run_test()
