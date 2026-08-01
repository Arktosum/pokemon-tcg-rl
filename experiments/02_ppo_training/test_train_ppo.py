import os
import sys
import torch
import csv
import glob

# Allow importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from train_ppo import log_to_csv, get_latest_checkpoint

def test_csv_logging(tmp_path):
    csv_file = tmp_path / "test_metrics.csv"
    
    metrics1 = {"timestamp": "2026-08-01 11:00:00", "update_count": 1, "win_rate": 50.0}
    metrics2 = {"timestamp": "2026-08-01 11:00:15", "update_count": 2, "win_rate": 75.0}
    
    log_to_csv(csv_file, metrics1)
    log_to_csv(csv_file, metrics2)
    
    assert os.path.exists(csv_file)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 2
    assert rows[0]["update_count"] == "1"
    assert rows[1]["win_rate"] == "75.0"

def test_checkpointing_fallback(tmp_path):
    # Mocking glob for get_latest_checkpoint inside the test
    # We can just create fake files in a temporary dir and monkeypatch os.path.dirname(__file__)
    import train_ppo
    
    original_dirname = os.path.dirname(train_ppo.__file__)
    train_ppo.__file__ = os.path.join(tmp_path, "train_ppo.py")
    
    # 1. No files
    ckpt, ctype = train_ppo.get_latest_checkpoint()
    assert ckpt is None
    assert ctype is None
    
    # 2. Only BC file
    bc_file = tmp_path / "20260731_214125_titan_bc.pt"
    bc_file.write_text("fake")
    ckpt, ctype = train_ppo.get_latest_checkpoint()
    assert ckpt == str(bc_file)
    assert ctype == "BC"
    
    # 3. PPO file exists
    ppo_file = tmp_path / "20260801_111111_ppo_checkpoint.pt"
    ppo_file.write_text("fake")
    ckpt, ctype = train_ppo.get_latest_checkpoint()
    assert ckpt == str(ppo_file)
    assert ctype == "PPO"
    
    # Restore
    train_ppo.__file__ = os.path.join(original_dirname, "train_ppo.py")
