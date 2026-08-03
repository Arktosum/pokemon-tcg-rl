import os
import torch
from pathlib import Path

def save_checkpoint(model, optimizer, episode, filename="latest.pt"):
    checkpoint_dir = Path(os.path.dirname(__file__)) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    filepath = checkpoint_dir / filename
    
    # Save only state dictionaries to avoid class definition coupling
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def load_checkpoint(model, optimizer=None, filename="latest.pt"):
    filepath = Path(os.path.dirname(__file__)) / "checkpoints" / filename
    if not filepath.exists():
        return False
        
    # Set weights_only=True for safe unpickling
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=True) 
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint.get('episode', 0)
