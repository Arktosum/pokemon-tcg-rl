import os

def generate_selfplay_script(checkpoint_filename="latest.pt"):
    base_dir = os.path.dirname(__file__).replace("\\", "/")
    script_path = os.path.join(base_dir, "temp_selfplay_agent.py")
    checkpoint_path = os.path.join(
        base_dir, "checkpoints", checkpoint_filename
    ).replace("\\", "/")
    
    # We write a self-contained script that the Kaggle engine can evaluate
    script_content = f"""
import sys
import os
import torch
import orjson
import numpy as np

# Force absolute path imports for the agent script
BASE_DIR = r"{base_dir}"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..", "01_baseline", "agent")))

from cg.api import to_observation_class
from parser import get_encoder_input, get_decoder_input
from main import read_deck_csv
from model_ppo import TitanTransformerPPO
from ppo_config import load_config

DECK_LIST = read_deck_csv()

# Initialize model once globally
titan_cfg, _, _ = load_config()
model = TitanTransformerPPO(titan_cfg)

checkpoint_path = r"{checkpoint_path}"
if os.path.exists(checkpoint_path):
    try:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model_state_dict'])
    except Exception as e:
        pass
model.eval()

def agent(obs, config):
    if obs is None: 
        return DECK_LIST
    
    try:
        obs_json = orjson.loads(obs) if isinstance(obs, str) else obs
        if isinstance(obs_json, list): obs_json = obs_json[0]
        if 'observation' in obs_json: obs_json = obs_json['observation']
        
        obs_class = to_observation_class(obs_json)
        if obs_class.select is None or obs_class.current is None: 
            return DECK_LIST
        
        enc_sv = get_encoder_input(obs_class)
        num_options = len(obs_class.select.option) if obs_class.select and obs_class.select.option else 1


        max_count = obs_class.select.maxCount if obs_class.select and hasattr(obs_class.select, 'maxCount') else 1
        actions = [[j] for j in range(num_options)]
        dec_sv = get_decoder_input(obs_class, actions)
        
        # Format tensors
        device = 'cpu'
        enc_indices = torch.as_tensor(np.array(enc_sv.index, dtype=np.int64), device=device).unsqueeze(0)
        enc_values = torch.as_tensor(np.array(enc_sv.value, dtype=np.float32), device=device).unsqueeze(0)
        enc_offsets = torch.as_tensor(np.array(enc_sv.offset, dtype=np.int64), device=device).unsqueeze(0)
        
        dec_idx = list(dec_sv.index)
        dec_val = list(dec_sv.value)
        dec_off = list(dec_sv.offset)
        
        dec_inputs_list = []
        for a in range(num_options):
            start = dec_off[a] if a < len(dec_off) else len(dec_idx)
            end = dec_off[a+1] if a+1 < len(dec_off) else len(dec_idx)
            idxs = torch.as_tensor(np.array(dec_idx[start:end], dtype=np.int64), device=device)
            vals = torch.as_tensor(np.array(dec_val[start:end], dtype=np.float32), device=device)
            dec_inputs_list.append((idxs, vals, start))
            
        action_masks = torch.zeros((1, num_options), dtype=torch.bool, device=device)
        
        with torch.no_grad():
            logits, _ = model(enc_indices, enc_values, enc_offsets, [dec_inputs_list], action_masks)
            masked_logits = logits.masked_fill(action_masks, float('-inf'))
            if max_count > 1 and max_count <= num_options:
                _, topk_actions = torch.topk(masked_logits, k=max_count, dim=-1)
                return topk_actions[0].cpu().tolist()
            else:
                return [int(torch.argmax(masked_logits, dim=-1).item())]
    except Exception:
        return [0]
"""
    with open(script_path, "w") as f:
        f.write(script_content)
    
    return script_path
