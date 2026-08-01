import pytest
import torch
import torch.nn as nn
from model import TitanTransformer

def test_model_forward_single_batch():
    """Verify that a single unbatched forward pass successfully computes logits across legal actions."""
    model = TitanTransformer(d_model=32, nhead=2, num_layers=1)
    
    enc_indices = torch.randint(0, 1000, (50,))
    enc_weights = torch.rand((50,))
    enc_offsets = torch.sort(torch.randint(0, 50, (24,)))[0]
    enc_offsets[0] = 0 
    
    dec_indices = torch.randint(0, 1000, (15,))
    dec_weights = torch.rand((15,))
    dec_offsets = torch.sort(torch.randint(0, 15, (5,)))[0]
    dec_offsets[0] = 0
    
    logits, value = model(enc_indices, enc_offsets, enc_weights, dec_indices, dec_offsets, dec_weights)
    
    assert logits.shape == (1, 5)
    assert value.shape == (1, 1)
    assert value.item() >= -1.0 and value.item() <= 1.0

def test_model_forward_dynamic_batching():
    """Verify that PPO mini-batching correctly pads and masks invalid actions with -inf."""
    model = TitanTransformer(d_model=32, nhead=2, num_layers=1)
    
    enc_indices = torch.randint(0, 1000, (100,))
    enc_weights = torch.rand((100,))
    enc_offsets = torch.sort(torch.randint(0, 100, (72,)))[0]
    enc_offsets[0] = 0
    
    batch_lengths = torch.tensor([2, 5, 1])
    
    dec_indices = torch.randint(0, 1000, (30,))
    dec_weights = torch.rand((30,))
    dec_offsets = torch.sort(torch.randint(0, 30, (8,)))[0]
    dec_offsets[0] = 0
    
    logits, value = model(
        enc_indices, enc_offsets, enc_weights, 
        dec_indices, dec_offsets, dec_weights, 
        batch_lengths=batch_lengths
    )
    
    assert logits.shape == (3, 5)
    
    assert torch.isneginf(logits[0, 2])
    assert not torch.isneginf(logits[0, 1])
    assert torch.isneginf(logits[2, 1])
    assert not torch.isneginf(logits[2, 0])
    assert not torch.isneginf(logits[1, 4])
    
    assert value.shape == (3, 1)
