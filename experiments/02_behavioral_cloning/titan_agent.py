import os
import sys
import torch

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = r"g:\programming\github-repositories\pokemon-tcg-rl\experiments\02_behavioral_cloning"
_baseline_agent_dir = os.path.abspath(os.path.join(_this_dir, "..", "01_baseline", "agent"))
_repo_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))

if _baseline_agent_dir not in sys.path:
    sys.path.insert(0, _baseline_agent_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from cg.api import to_observation_class, Observation, LogType
from parser import get_encoder_input, get_decoder_input
from main import read_deck_csv
from model import TitanTransformer, TitanConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model():
    global model
    if model is None:
        cfg = TitanConfig()
        model = TitanTransformer(cfg)
        
        # Determine checkpoint path
        ckpt_path = os.path.join(_this_dir, "checkpoints", "titan_bc_best.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = "/kaggle_simulations/agent/checkpoints/titan_bc_best.pt"
            
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()

def agent(obs_dict: dict) -> list[int]:
    """
    Kaggle-compatible inference wrapper for TitanTransformer Behavioral Cloning agent.
    """
    # 1. Turn-0 Deck Loading Trap
    if obs_dict.get('select') is None:
        return read_deck_csv()
        
    obs: Observation = to_observation_class(obs_dict)
    
    legal_option_count = len(obs.select.option)
    if legal_option_count == 0:
        return []
    if legal_option_count == 1:
        return [0]
        
    # 2. Model Loading
    load_model()
    
    # 3. Observation Parsing
    sv_enc = get_encoder_input(obs)
    
    enc_indices = torch.tensor([sv_enc.index], dtype=torch.long, device=device)
    enc_values = torch.tensor([sv_enc.value], dtype=torch.float32, device=device)
    enc_offsets = torch.tensor([sv_enc.offset], dtype=torch.long, device=device)
    
    actions = [[i] for i in range(legal_option_count)]
    sv_dec = get_decoder_input(obs, actions)
    
    cfg = model.config
    # Calculate padded length
    N = max(cfg.max_actions, legal_option_count)
    
    offset_list = list(sv_dec.offset)
    while len(offset_list) < N:
        offset_list.append(len(sv_dec.index))
        
    dec_indices = torch.tensor(sv_dec.index, dtype=torch.long, device=device)
    dec_values = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
    dec_offsets = torch.tensor(offset_list, dtype=torch.long, device=device)
    
    # 5. Action Masking: padding set to -1e9 inside the model forward via action_mask
    action_mask = torch.zeros((1, N), dtype=torch.bool, device=device)
    if legal_option_count < N:
        action_mask[0, legal_option_count:] = True
        
    # 4. Forward Pass
    with torch.no_grad():
        logits = model(
            enc_indices=enc_indices,
            enc_values=enc_values,
            enc_offsets=enc_offsets,
            dec_indices=dec_indices,
            dec_values=dec_values,
            dec_offsets=dec_offsets,
            action_mask=action_mask
        )
        
    # 6. Return
    best_action_idx = torch.argmax(logits[0]).item()
    result_action = [best_action_idx]

    turn = obs_dict.get('current', {}).get('turn', 0)
    for log in obs.logs:
        if log.type == LogType.RESULT:
            print(f"[TURN {turn}] MATCH ENDED. REASON: {log.reason}")

    if turn <= 2:
        print(f"--- TELEMETRY Turn {turn} ---")
        print(f"SelectContext: {obs.select.context}")
        print(f"minCount: {obs.select.minCount}, maxCount: {obs.select.maxCount}")
        print(f"Valid options count: {legal_option_count}")
        print(f"Raw logits: {logits[0].cpu().numpy().tolist()}")
        print(f"Final returned action: {result_action}")
        print("---------------------------")

    return result_action
