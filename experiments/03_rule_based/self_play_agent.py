import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import glob
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent')))
from cg.api import to_observation_class
from parser import get_encoder_input, get_decoder_input

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02_ppo_training')))
from model import TitanTransformer

# Global state to keep the model loaded in the worker process
_MODEL = None
_DEVICE = torch.device("cpu")

def _get_latest_checkpoint():
    ppo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02_ppo_training'))
    ppo_checkpoints = glob.glob(os.path.join(ppo_dir, "*_ppo_checkpoint.pt"))
    if ppo_checkpoints:
        ppo_checkpoints.sort(key=os.path.getmtime)
        return ppo_checkpoints[-1]
    return None

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
        
    _MODEL = TitanTransformer().to(_DEVICE)
    _MODEL.eval()
    
    ckpt_path = _get_latest_checkpoint()
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=_DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            _MODEL.load_state_dict(checkpoint["model_state_dict"])
        else:
            _MODEL.load_state_dict(checkpoint, strict=False)
            
    return _MODEL

def _read_deck_csv() -> list[int]:
    file_path = os.path.join(os.path.dirname(__file__), "..", "01_baseline", "agent", "deck.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    
    try:
        with open(file_path, "r") as file:
            csv = file.read().split("\n")
        deck = []
        for i in range(60):
            if csv[i].strip():
                deck.append(int(csv[i]))
        return deck
    except Exception:
        return [1] * 60

def self_play_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    deck = _read_deck_csv()
    
    if obs.select is None:
        return deck
        
    options = obs.select.option
    max_count = obs.select.maxCount
    min_count = obs.select.minCount
    if max_count == 0 or not options:
        return []
    
    # CRITICAL: If multi-select is required, the model can only score single actions.
    # Fall back to random selection for multi-select contexts to avoid instant forfeit.
    if min_count > 1:
        import random
        count = min(max_count, len(options))
        count = max(count, min_count)
        count = min(count, len(options))
        return random.sample(list(range(len(options))), count)
        
    model = _load_model()
    
    sv_enc = get_encoder_input(obs, deck)
    
    legal_count_val = len(options)
    actions = [[i] for i in range(legal_count_val)]
    sv_dec = get_decoder_input(obs, actions)
    
    enc_indices = torch.tensor(sv_enc.index, dtype=torch.int32, device=_DEVICE)
    enc_offsets = torch.tensor(sv_enc.offset, dtype=torch.int32, device=_DEVICE)
    enc_weights = torch.tensor(sv_enc.value, dtype=torch.float32, device=_DEVICE)
    
    dec_indices = torch.tensor(sv_dec.index, dtype=torch.int32, device=_DEVICE)
    dec_offsets = torch.tensor(sv_dec.offset, dtype=torch.int32, device=_DEVICE)
    dec_weights = torch.tensor(sv_dec.value, dtype=torch.float32, device=_DEVICE)
    
    legal_count = torch.tensor([len(sv_dec.offset)], dtype=torch.int32, device=_DEVICE)
    
    with torch.no_grad():
        logits, _ = model(
            enc_indices, enc_offsets, enc_weights,
            dec_indices, dec_offsets, dec_weights,
            legal_count
        )
        
        probs = F.softmax(logits, dim=-1)
        # We can either sample or take argmax. Self-play usually works better with some stochasticity or argmax?
        # Let's take argmax for self-play opponent to be strong and consistent.
        action_idx = torch.argmax(probs).item()
        
    return [action_idx]
