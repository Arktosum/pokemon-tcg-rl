import sys
import os
import pickle
from tqdm import tqdm

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)
sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from cg.game import battle_start, battle_select
from cg.api import to_observation_class
from src.model.transformer_policy import get_encoder_input, get_decoder_input
from src.model.heuristic_bot import agent as rule_agent

def generate_dataset(num_games=5, output_file='input/bc_dataset.pkl'):
    dataset = []
    
    # Using the standard Kaggle mega lucario/crustle deck (deck ID 0 usually, but let's just use the sample one)
    deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    
    print(f"Generating {num_games} games of heuristic self-play for BC...")
    
    for game_idx in tqdm(range(num_games)):
        obs_dict, start_data = battle_start(deck, deck)
        
        while True:
            if not obs_dict:
                break # Game over
                
            obs = to_observation_class(obs_dict)
            
            # 1. Get the PyTorch Network Inputs
            sv_enc = get_encoder_input(obs, deck)
            legal_actions = [[i] for i in range(len(obs.select.option))]
            sv_dec = get_decoder_input(obs, legal_actions)
            
            # 2. Get the Heuristic Bot's Chosen Action
            try:
                chosen_action = rule_agent(obs_dict)
            except Exception as e:
                print(f"Heuristic bot crashed: {e}")
                break
                
            # 3. Find the target index
            target_idx = legal_actions.index(chosen_action) if chosen_action in legal_actions else -1
            
            if target_idx != -1:
                # 4. Save to dataset
                dataset.append({
                    'sv_enc': sv_enc,
                    'sv_dec': sv_dec,
                    'target_idx': target_idx
                })
            
            # Step the environment
            try:
                obs_dict = battle_select(chosen_action)
            except Exception:
                # battle_ptr broken or IndexError (game ended abruptly)
                break
                
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"Successfully generated BC dataset with {len(dataset)} state-action pairs.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
