import sys
import os

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)
sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from cg.game import battle_start, battle_select
from cg.api import to_observation_class
from src.model.transformer_policy import get_encoder_input, get_decoder_input
from src.model.heuristic_bot import agent as rule_agent

def test_generation():
    deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    
    obs_dict, start_data = battle_start(deck, deck)
    
    for step_num in range(10):
        if not obs_dict:
            print("Game over.")
            break
            
        obs = to_observation_class(obs_dict)
        
        sv_enc = get_encoder_input(obs, deck)
        legal_actions = [[i] for i in range(len(obs.select.option))]
        sv_dec = get_decoder_input(obs, legal_actions)
        
        try:
            print(f"\n--- Step {step_num} ---")
            print(f"Legal Actions: {len(legal_actions)}")
            
            chosen_action = rule_agent(obs_dict)
            print(f"Heuristic chose action: {chosen_action}")
            
            obs_dict = battle_select(chosen_action)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    test_generation()
