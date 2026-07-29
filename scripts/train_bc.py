import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import time
import copy

# Ensure we can import src
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from src.model import PokemonActorCritic

def train_bc():
    train_path = REPO_ROOT / "data" / "bc_train_full.pt"
    val_path = REPO_ROOT / "data" / "bc_val_full.pt"
    
    if not train_path.exists() or not val_path.exists():
        print(f"Error: Dataset not found.")
        sys.exit(1)
        
    print(f"Loading datasets...", flush=True)
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    
    train_dataset = TensorDataset(train_data["states"], train_data["actions"])
    val_dataset = TensorDataset(val_data["states"], val_data["actions"])
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    model = PokemonActorCritic().to(device)
    
    # Freeze the value head since BC only trains the policy
    for param in model.value_head.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    epochs = 100
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Starting BC training for max {epochs} epochs with Early Stopping (patience={patience})...", flush=True)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_total = 0
        
        for batch_states, batch_actions in train_loader:
            batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)
            
            optimizer.zero_grad()
            
            policy_probs, _ = model(batch_states)
            action_logits = torch.log(policy_probs + 1e-8)
            
            loss = criterion(action_logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_states.size(0)
            train_total += batch_states.size(0)
            
        epoch_train_loss = train_loss / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)
                
                policy_probs, _ = model(batch_states)
                action_logits = torch.log(policy_probs + 1e-8)
                
                loss = criterion(action_logits, batch_actions)
                
                val_loss += loss.item() * batch_states.size(0)
                
                _, predicted = torch.max(policy_probs, 1)
                val_correct += (predicted == batch_actions).sum().item()
                val_total += batch_states.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        exec_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:03d} | Time: {exec_time:5.1f}s | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}", flush=True)
        
        # Early Stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs! Best Val Loss: {best_val_loss:.4f}", flush=True)
                break
                
        scheduler.step()
                
    # Save best model
    checkpoints_dir = REPO_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    out_path = checkpoints_dir / "TOP_ELO_BC_MODEL_FINAL.pt"
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), out_path)
    print(f"Best model saved to {out_path}")

if __name__ == "__main__":
    train_bc()
