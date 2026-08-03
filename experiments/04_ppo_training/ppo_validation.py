import torch
import torch.nn as nn
from dataclasses import dataclass
from ppo_env import PokemonPPOEnv

import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "01_baseline", "agent")))
from main import read_deck_csv
DECK_LIST = read_deck_csv()

@dataclass
class ValidationResult:
    win_rate: float
    total_games: int

def run_validation(model: nn.Module, sampler, num_games: int = 5, device: str = "cpu") -> ValidationResult:
    model.eval()
    wins = 0
    games_played = 0
    
    for game_idx in range(num_games):
        opponent_path, opponent_name = sampler.sample_opponent()
        env = PokemonPPOEnv(opponent=opponent_path)
        step_result = env.reset()
        done = step_result.done
        
        while not done:
            parsed = step_result.obs
            if parsed is None:
                step_result = env.step(DECK_LIST)
                if step_result.info.get('is_invalid'):
                    print(f"FATAL ENGINE REJECTION (SETUP): Attempted Action: {step_result.info.get('failed_action')}")
                    raise RuntimeError("INVALID step! Action failed engine check during setup.")
                done = step_result.done
                continue

            num_options = parsed.num_options
            max_count = parsed.max_count

            # Add batch dimension
            enc_indices = torch.tensor(parsed.enc_index, dtype=torch.long, device=device).unsqueeze(0)
            enc_values = torch.tensor(parsed.enc_value, dtype=torch.float, device=device).unsqueeze(0)
            enc_offsets = torch.tensor(parsed.enc_offset, dtype=torch.long, device=device).unsqueeze(0)

            dec_inputs_list = []
            for a in range(num_options):
                start = parsed.dec_offset[a] if a < len(parsed.dec_offset) else len(parsed.dec_index)
                end = parsed.dec_offset[a+1] if a+1 < len(parsed.dec_offset) else len(parsed.dec_index)
                
                idxs = torch.tensor(parsed.dec_index[start:end], dtype=torch.long, device=device)
                vals = torch.tensor(parsed.dec_value[start:end], dtype=torch.float, device=device)
                dec_inputs_list.append((idxs, vals, start))

            action_masks = torch.zeros((1, num_options), dtype=torch.bool, device=device)

            with torch.no_grad():
                logits, _val = model(enc_indices, enc_values, enc_offsets, [dec_inputs_list], action_masks)
                
                masked_logits = logits.masked_fill(action_masks, float('-inf'))
                
                # Multi-action top-k selection
                if max_count > 1 and max_count <= num_options:
                    _, topk_actions = torch.topk(masked_logits, k=max_count, dim=-1)
                    action = topk_actions[0].cpu().tolist()
                else:
                    action = [int(torch.argmax(masked_logits, dim=-1).item())]
            
            step_result = env.step(action)
            done = step_result.done
        
        # Kaggle reward for win is > 0
        if step_result.reward > 0:
            wins += 1
        games_played += 1
        
        # SAVE REPLAY LOGIC (Only save the first game of the validation batch to avoid clutter)
        if game_idx == 0:
            replay_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "replays"))
            os.makedirs(replay_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(replay_dir, f"val_replay_{timestamp}.json")
            
            # Extract Kaggle's internal JSON match record
            try:
                match_data = env.env.toJSON()
                if "info" not in match_data:
                    match_data["info"] = {}
                match_data["info"]["TeamNames"] = ["Titan PPO", opponent_name]
                
                with open(json_path, "w") as f:
                    json.dump(match_data, f)
                print(f"Saved validation replay to: {json_path}")
                
                # Attempt to trigger the MD parser if it exists in the baseline/tools
                md_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "replay_to_md.py"))
                os.system(f"python {md_script} {json_path}")
            except Exception as e:
                print(f"Failed to save replay: {e}")
    model.train()
    return ValidationResult(win_rate=wins/max(1, games_played), total_games=games_played)
