import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train_data import ReplayDataset, collate_fn
from agent.model import MyModel

def train():
    print("Initializing Supervised Training Pipeline...")
    
    # 1. Dataset and DataLoader
    # Using batch_size=1 to avoid padding variable-length action spaces.
    # We will use gradient accumulation to simulate a larger batch size.
    replay_dir = os.path.dirname(__file__) # Load from the current dir (replay_20260730_125951.json)
    dataset = ReplayDataset(replay_dir)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    # 2. Model and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MyModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # 3. Loss Functions
    # CrossEntropy expects raw logits, not tanh. MyModel outputs tanh for policy. 
    # Since it's tanh bounded (-1, 1), CrossEntropy can still process it, 
    # but it's mathematically better to use MSE if it's already bounded, OR just use CrossEntropy.
    # For now, we'll use CrossEntropy Loss for Policy.
    ce_loss = nn.CrossEntropyLoss()
    huber_loss = nn.HuberLoss()
    
    accumulation_steps = 16
    epochs = 10
    
    print("\n--- STARTING TRAINING ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        steps = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            (enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off, target_action, target_value) = batch
            
            # Move to device
            enc_idx, enc_val, enc_off = enc_idx.to(device), enc_val.to(device), enc_off.to(device)
            dec_idx, dec_val, dec_off = dec_idx.to(device), dec_val.to(device), dec_off.to(device)
            target_action, target_value = target_action.to(device), target_value.to(device)
            
            # Forward pass
            # MyModel outputs v: [batch_size, 1], p: [batch_size, num_actions]
            v, p = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
            
            # Calculate Loss
            # Policy is [batch_size, num_actions], target_action is [batch_size]
            loss_p = ce_loss(p, target_action)
            # Value is [batch_size, 1], target_value is [batch_size, 1]
            loss_v = huber_loss(v, target_value)
            
            loss = loss_p + loss_v
            
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            total_policy_loss += loss_p.item()
            total_value_loss += loss_v.item()
            steps += 1
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        # Handle remaining gradients
        optimizer.step()
        optimizer.zero_grad()
        
        if steps > 0:
            avg_loss = total_loss / steps
            avg_p_loss = total_policy_loss / steps
            avg_v_loss = total_value_loss / steps
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} | No data processed.")

if __name__ == '__main__':
    train()
