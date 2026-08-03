import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ppo_types import ParsedObs
from ppo_memory import RolloutBuffer

def test_rollout_buffer_jagged_padding():
    buffer = RolloutBuffer(device="cpu")
    
    # Step 1: short enc, 2 options
    obs1 = ParsedObs(
        enc_index=[1, 2], enc_value=[1.0, 1.0], enc_offset=[0, 2],
        dec_index=[3, 4], dec_value=[1.0, 1.0], dec_offset=[0, 1, 2],
        num_options=2, max_count=1
    )
    buffer.add(obs1, action=[1], reward=0.0, done=False, value=0.5, log_prob=-0.6)
    
    # Step 2: longer enc, 3 options
    obs2 = ParsedObs(
        enc_index=[1, 2, 3, 4], enc_value=[1.0, 1.0, 1.0, 1.0], enc_offset=[0, 2, 4],
        dec_index=[3, 4, 5], dec_value=[1.0, 1.0, 1.0], dec_offset=[0, 1, 2, 3],
        num_options=3, max_count=1
    )
    buffer.add(obs2, action=[2], reward=1.0, done=True, value=0.8, log_prob=-0.2)
    
    batch = buffer.get_batch()
    
    assert batch.enc_indices.shape == (2, 4), "Should pad enc_indices to max length 4"
    assert batch.action_masks.shape == (2, 3), "Should pad action_masks to max options 3"
    assert batch.actions.shape == (2, 1), "Actions shape should match max_count or list len"
    
    # First step should have True (invalid) for the 3rd option since it only had 2 options
    assert batch.action_masks[0, 2].item() == True
    assert batch.action_masks[0, 0].item() == False
    
    # Second step should have False for all 3 options
    assert batch.action_masks[1, 2].item() == False
    
    buffer.clear()
    assert len(buffer.actions) == 0, "Buffer should be cleared"
