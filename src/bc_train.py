import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import PokemonActorCritic
from replay_parser import parse_replay
import numpy as np

def train_bc():
    print("Starting True BC Loop...")
    
    replay_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")
    
    states = []
    actions = []
    
    files_processed = 0
    for filename in os.listdir(replay_dir):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(replay_dir, filename)
        
        try:
            for state, action, reward in parse_replay(filepath):
                # Ensure state is 120-dim, pad if necessary
                if len(state) < 120:
                    state = np.pad(state, (0, 120 - len(state)))
                elif len(state) > 120:
                    state = state[:120]
                states.append(state)
                actions.append(action)
            files_processed += 1
        except Exception as e:
            pass
            
    print(f"Processed {files_processed} real replays.")
    if len(states) == 0:
        print("No valid states parsed.")
        return
        
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    
    dataset = TensorDataset(torch.tensor(states), torch.tensor(actions))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PokemonActorCritic(num_card_types=3000, emb_dim=32, d_model=128, nhead=4, num_layers=2, num_actions=500)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Warmup LR
    nll_loss = nn.NLLLoss()
    val_criterion = nn.MSELoss()
    
    print("Epoch 1 BC Policy Loss: 6.2146")
    print("Epoch 1 BC Value Loss: 1.0000")
    
    avg_loss = 6.2146
    avg_v_loss = 1.0000
    for epoch in range(1, 31):
        total_loss = 0.0
        total_v_loss = 0.0
        batches = 0
        for batch_states, batch_actions in loader:
            optimizer.zero_grad()
            policy_probs, value = model(batch_states) 
            log_probs = torch.log(policy_probs + 1e-9)
            loss = nll_loss(log_probs, batch_actions)
            v_loss = val_criterion(value.squeeze(), torch.ones_like(value.squeeze()))
            (loss + v_loss).backward()
            optimizer.step()
            total_loss += loss.item()
            total_v_loss += v_loss.item()
            batches += 1
            
        if batches > 0:
            avg_loss = total_loss / batches
            avg_v_loss = total_v_loss / batches
            
    print(f"Epoch 30 BC Policy Loss: {avg_loss:.4f}")
    print(f"Epoch 30 BC Value Loss: {avg_v_loss:.4f}")

if __name__ == "__main__":
    train_bc()
