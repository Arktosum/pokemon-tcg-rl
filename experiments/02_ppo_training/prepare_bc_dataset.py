import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Any

# Ensure we can import the baseline parser
baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline'))
if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from agent.cg.api import to_observation_class, Observation
from agent.parser import get_encoder_input, get_decoder_input, SparseVector

def process_replay(replay_path: str) -> List[Dict[str, Any]]:
    """
    Parses a single Kaggle replay JSON and extracts Behavioral Cloning samples for the winning agent.
    Returns a list of samples. Each sample contains:
    - 'encoder_indices': List[int]
    - 'encoder_values': List[float]
    - 'encoder_offsets': List[int]
    - 'decoder_indices': List[int]
    - 'decoder_values': List[float]
    - 'decoder_offsets': List[int]
    - 'target_actions': List[int]
    """
    with open(replay_path, 'r', encoding='utf-8') as f:
        try:
            replay = json.load(f)
        except json.JSONDecodeError:
            return []

    # 1. Identify the winner
    rewards = replay.get('rewards', [])
    if not rewards or max(rewards) <= 0:
        return [] # Draw or everyone lost/errored
    
    winner_index = rewards.index(max(rewards))

    samples = []
    steps = replay.get('steps', [])
    
    # 2. Extract observations
    for step_idx, step_data in enumerate(steps):
        # step_data is a list of agent dicts
        if len(step_data) <= winner_index:
            continue
            
        agent_data = step_data[winner_index]
        obs_dict = agent_data.get('observation')
        action = agent_data.get('action') # List of ints chosen by the agent
        
        if not obs_dict or action is None:
            continue
            
        try:
            obs: Observation = to_observation_class(obs_dict)
        except Exception as e:
            # Engine parsing failure on a corrupted step
            continue
            
        # We only care about steps where the agent had to select options via the Pointer Network.
        # Initial deck selection (select == None) is ignored.
        if obs.select is None:
            continue
            
        # 3. Vectorize the board state (Encoder)
        try:
            sv_enc = get_encoder_input(obs)
            
            # 4. Vectorize the legal actions (Decoder)
            legal_option_count = len(obs.select.option)
            if legal_option_count == 0:
                continue
                
            # Kaggle Competitor Bug Mitigation: Competitors often return invalid actions or 
            # deck arrays (len == 60) out-of-sync. If the target index exceeds legal options, skip it.
            if len(action) == 0 or len(action) == 60:
                continue
            
            target_idx = action[0]
            if target_idx < 0 or target_idx >= legal_option_count:
                continue
                
            # Simulate querying the parser for every individual legal option
            all_legal_actions = [[i] for i in range(legal_option_count)]
            sv_dec = get_decoder_input(obs, all_legal_actions)
        except Exception as e:
            # In case the parser throws an error on a weird edge case, we skip this step
            continue

        sample = {
            'encoder_indices': sv_enc.index,
            'encoder_values': sv_enc.value,
            'encoder_offsets': sv_enc.offset,
            'decoder_indices': sv_dec.index,
            'decoder_values': sv_dec.value,
            'decoder_offsets': sv_dec.offset,
            'target_actions': action, # e.g. [4, 2]
            'legal_option_count': legal_option_count
        }
        samples.append(sample)
        
    return samples

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_directory(dataset_dir: str, output_file: str, max_files: int = None):
    dataset_path = Path(dataset_dir)
    json_files = list(dataset_path.glob("*.json"))
    
    if max_files:
        json_files = json_files[:max_files]
        
    print(f"Found {len(json_files)} replays to process.")
    
    cores = cpu_count()
    print(f"Booting up multiprocessing pool with {cores} cores...")
    
    # Use multiprocessing to parse JSONs and tensors in parallel
    with Pool(processes=cores) as pool:
        # pool.imap allows us to use tqdm for a progress bar
        results = list(tqdm(pool.imap(process_replay, [str(p) for p in json_files]), total=len(json_files)))
        
    all_samples = [item for sublist in results for item in sublist]
        
    print(f"Total samples extracted: {len(all_samples)}")
    print(f"Saving to {output_file}...")
    torch.save(all_samples, output_file)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../01_baseline/dataset/matches")
    parser.add_argument("--output_file", type=str, default="bc_dataset.pt")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()
    
    process_directory(args.dataset_dir, args.output_file, args.max_files)
