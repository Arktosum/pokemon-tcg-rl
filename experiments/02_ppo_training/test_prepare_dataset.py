import os
import json
import tempfile
import pytest
from pathlib import Path

from prepare_bc_dataset import process_replay

# Find a real replay file to test against
DATASET_DIR = Path(os.path.abspath(__file__)).parent.parent / "01_baseline" / "dataset" / "matches"
REAL_REPLAY_PATH = None
if DATASET_DIR.exists():
    json_files = list(DATASET_DIR.glob("*.json"))
    if json_files:
        REAL_REPLAY_PATH = str(json_files[0])

@pytest.mark.skipif(not REAL_REPLAY_PATH, reason="No real replay files found in dataset directory.")
def test_process_replay_winner_extraction():
    """Verifies that the script correctly extracts actions for the winning agent using a real replay."""
    samples = process_replay(REAL_REPLAY_PATH)
    
    # A real replay should yield numerous samples (one for every turn the winner acted)
    assert len(samples) > 0, "Failed to extract any samples from a real replay file."
    
    sample = samples[0]
    
    # Verify the dictionary structure
    assert "encoder_indices" in sample
    assert "decoder_indices" in sample
    assert "target_actions" in sample
    assert "encoder_offsets" in sample
    assert "decoder_offsets" in sample
    assert "legal_option_count" in sample
    
    # Verify mathematical shape of our architecture (Encoder must have 24 words/offsets)
    assert len(sample["encoder_offsets"]) == 24, "Encoder must output exactly 24 words."
    
    # Target actions should be a list (even if empty)
    assert isinstance(sample["target_actions"], list)

@pytest.mark.skipif(not REAL_REPLAY_PATH, reason="No real replay files found in dataset directory.")
def test_process_replay_draw():
    """Verifies that replays resulting in a draw or timeout are skipped."""
    with open(REAL_REPLAY_PATH, 'r') as f:
        real_replay = json.load(f)
        
    # Force a draw
    real_replay['rewards'] = [0, 0]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(real_replay, f)
        temp_path = f.name
        
    try:
        samples = process_replay(temp_path)
        # Should return an empty list because max(rewards) == 0
        assert len(samples) == 0, "Replays with draws should be ignored."
    finally:
        os.remove(temp_path)

def test_corrupted_json():
    """Verifies that the script gracefully skips corrupted JSON files without crashing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{invalid_json:")
        temp_path = f.name
        
    try:
        samples = process_replay(temp_path)
        assert len(samples) == 0, "Corrupted JSON should yield 0 samples."
    finally:
        os.remove(temp_path)
