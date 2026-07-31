import os
import json
import glob
import torch
from torch.utils.data import IterableDataset, DataLoader
from agent.parser import get_encoder_input, get_decoder_input
from agent.cg.api import to_observation_class

class ReplayDataset(IterableDataset):
    def __init__(self, replay_dir):
        super().__init__()
        self.replay_files = glob.glob(os.path.join(replay_dir, "replay_*.json"))
        
    def __iter__(self):
        for replay_file in self.replay_files:
            with open(replay_file, 'r', encoding='utf-8') as f:
                try:
                    replay = json.load(f)
                except Exception as e:
                    print(f"Error loading {replay_file}: {e}")
                    continue
            
            # Determine the winner from the final step rewards
            # Kaggle rewards: [1, -1] means P0 won, [-1, 1] means P1 won, [0, 0] means draw/unfinished
            final_rewards = [s['reward'] for s in replay['steps'][-1]]
            winner = None
            if final_rewards[0] == 1 and final_rewards[1] == -1:
                winner = 0
            elif final_rewards[0] == -1 and final_rewards[1] == 1:
                winner = 1
                
            for step_data in replay['steps']:
                for player_idx, player_data in enumerate(step_data):
                    # We only care about steps where a player took an action from a select prompt
                    if 'observation' not in player_data: continue
                    obs_dict = player_data['observation']
                    obs = to_observation_class(obs_dict)
                    
                    if not obs.select or len(obs.select.option) == 0:
                        continue # No choices to make
                        
                    # Extract the human's action
                    action_list = player_data.get('action', [])
                    if action_list is None: action_list = []
                    
                    if len(action_list) > 0:
                        target_action_idx = action_list[0]
                    else:
                        target_action_idx = 0 # Default/fallback if empty
                        
                    # Ensure target is within bounds (should always be true for valid replays)
                    if target_action_idx >= len(obs.select.option):
                        target_action_idx = 0
                        
                    # Target value
                    if winner is None:
                        target_value = 0.0 # Draw
                    elif winner == player_idx:
                        target_value = 1.0 # Win
                    else:
                        target_value = -1.0 # Loss
                        
                    # Parse Board (Encoder)
                    sv_enc = get_encoder_input(obs)
                    enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32)
                    enc_val = torch.tensor(sv_enc.value, dtype=torch.float32)
                    enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32)
                    
                    # Parse Actions (Decoder)
                    actions_list_of_lists = [[i] for i in range(len(obs.select.option))]
                    sv_dec = get_decoder_input(obs, actions_list_of_lists)
                    dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32)
                    dec_val = torch.tensor(sv_dec.value, dtype=torch.float32)
                    dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32)
                    
                    yield (enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off, target_action_idx, target_value)

# Custom collate function to handle variable-sized sparse vectors in a batch
def collate_fn(batch):
    # Batch is a list of tuples
    batch_enc_idx = []
    batch_enc_val = []
    batch_enc_off = []
    
    batch_dec_idx = []
    batch_dec_val = []
    batch_dec_off = []
    
    batch_target_action = []
    batch_target_value = []
    
    enc_word_count = 24
    
    enc_pos_offset = 0
    dec_pos_offset = 0
    
    for (enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off, target_action, target_value) in batch:
        batch_enc_idx.append(enc_idx)
        batch_enc_val.append(enc_val)
        # Offsets need to be shifted by the current position across the batch?
        # Actually EmbeddingBag offset is within the batch's concatenated indices.
        batch_enc_off.append(enc_off + enc_pos_offset)
        enc_pos_offset += len(enc_idx)
        
        batch_dec_idx.append(dec_idx)
        batch_dec_val.append(dec_val)
        batch_dec_off.append(dec_off + dec_pos_offset)
        dec_pos_offset += len(dec_idx)
        
        batch_target_action.append(target_action)
        batch_target_value.append(target_value)
        
    return (
        torch.cat(batch_enc_idx), torch.cat(batch_enc_val), torch.cat(batch_enc_off),
        torch.cat(batch_dec_idx), torch.cat(batch_dec_val), torch.cat(batch_dec_off),
        torch.tensor(batch_target_action, dtype=torch.long),
        torch.tensor(batch_target_value, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]
    )
