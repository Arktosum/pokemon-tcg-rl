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
    
    train_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bc_train_full.pt")
    val_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bc_val_full.pt")
    
    if not os.path.exists(train_path):
        print(f"Error: Preprocessed dataset not found at {train_path}")
        return
        
    print(f"Loading {train_path}...")
    train_data = torch.load(train_path)
    train_states = train_data["states"]
    train_actions = train_data["actions"]
    
    print(f"Loading {val_path}...")
    val_data = torch.load(val_path)
    val_states = val_data["states"]
    val_actions = val_data["actions"]
    
    train_dataset = TensorDataset(train_states, train_actions)
    val_dataset = TensorDataset(val_states, val_actions)
    
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} train pairs and {len(val_dataset)} val pairs.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = PokemonActorCritic(num_card_types=3000, emb_dim=32, d_model=128, nhead=4, num_layers=2, num_actions=500).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Warmup LR
    nll_loss = nn.NLLLoss()
    val_criterion = nn.MSELoss()
    
    num_epochs = 30
    total_train_batches = len(loader)
    
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TOP_ELO_BC_MODEL_FINAL.pt")
    
    # Resume from checkpoint if it exists
    if os.path.exists(model_path):
        print(f"Resuming from existing checkpoint: {model_path}", flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    for epoch in range(6, num_epochs + 1): # Start at 6 since we did 5
        model.train()
        total_loss = 0.0
        total_v_loss = 0.0
        correct_train = 0
        total_train = 0
        batches = 0
        for batch_idx, (batch_states, batch_actions) in enumerate(loader):
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            optimizer.zero_grad()
            policy_probs, value = model(batch_states) 
            log_probs = torch.log(policy_probs + 1e-9)
            loss = nll_loss(log_probs, batch_actions)
            v_loss = val_criterion(value.squeeze(), torch.ones_like(value.squeeze()))
            (loss + v_loss).backward()
            optimizer.step()
            total_loss += loss.item()
            total_v_loss += v_loss.item()
            
            preds = torch.argmax(policy_probs, dim=1)
            correct_train += (preds == batch_actions).sum().item()
            total_train += batch_actions.size(0)
            
            batches += 1
            
            if batches % 1000 == 0:
                print(f"Epoch {epoch} | Batch {batches}/{total_train_batches} | Loss: {loss.item():.4f} | Acc: {(correct_train/total_train)*100:.2f}%", flush=True)
            
        avg_loss = total_loss / max(1, batches)
        train_acc = (correct_train / total_train) * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                policy_probs, _ = model(batch_states)
                log_probs = torch.log(policy_probs + 1e-9)
                loss = nll_loss(log_probs, batch_actions)
                val_loss += loss.item()
                val_batches += 1
                
                preds = torch.argmax(policy_probs, dim=1)
                correct_val += (preds == batch_actions).sum().item()
                total_val += batch_actions.size(0)
                
        avg_val_loss = val_loss / max(1, val_batches)
        val_acc = (correct_val / total_val) * 100
            
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.2f}%)", flush=True)
        
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Saved checkpoint to {model_path}", flush=True)
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience {patience_counter}/{patience}", flush=True)
            if patience_counter >= patience:
                print("Early stopping triggered!", flush=True)
                break

if __name__ == "__main__":
    train_bc()
