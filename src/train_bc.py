import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)
sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from src.model.transformer_policy import MyModel

def train():
    dataset_path = 'input/bc_dataset.pkl'
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return
        
    print(f"Loading {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
        
    print(f"Loaded {len(dataset)} samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        
        for idx, sample in enumerate(dataset):
            sv_enc = sample['sv_enc']
            sv_dec = sample['sv_dec']
            target = sample['target_idx']
            
            # Skip invalid targets (e.g. game crashed or actions didn't match)
            if target == -1:
                continue
                
            enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32, device=device)
            enc_val = torch.tensor(sv_enc.value, dtype=torch.float32, device=device)
            enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32, device=device)
            
            dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32, device=device)
            dec_val = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
            dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
            
            optimizer.zero_grad()
            
            # Forward pass (batch_size = 1)
            value, policy_logits = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
            
            # policy_logits has shape [1, num_legal_actions]
            # Actually, `model()` returns policy of shape [1, num_legal_actions]?
            # Let's check model() return shape.
            # In `transformer_policy.py`: policy = self.decoder_fc(decoder_out).view(1, -1)
            # So it returns [1, N]. CrossEntropyLoss expects [batch_size, num_classes] and target [batch_size].
            
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)
            
            loss = criterion(policy_logits, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = policy_logits.argmax(dim=1).item()
            if predicted == target:
                correct += 1
                
        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
        
    os.makedirs('checkpoints', exist_ok=True)
    save_path = os.path.join('checkpoints', 'model_bc.pt')
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"Saved BC Pre-trained model to {save_path}")

if __name__ == "__main__":
    train()
