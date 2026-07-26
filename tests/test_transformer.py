import sys
import os
import torch

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)

sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from cg.game import battle_start
from cg.api import to_observation_class
from src.model.transformer_policy import MyModel, get_encoder_input, get_decoder_input, eval_nn

def test_model():
    print("Instantiating Transformer model...")
    model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1)
    
    print("Initializing cg engine to get dummy Observation...")
    sample_deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    obs_dict, _ = battle_start(sample_deck, sample_deck)
    obs = to_observation_class(obs_dict)
    
    print("Encoding state vectors...")
    sv_enc = get_encoder_input(obs, sample_deck)
    actions = [[i] for i in range(len(obs.select.option))]
    sv_dec = get_decoder_input(obs, actions)
    
    print("Running forward pass...")
    value, policy = eval_nn(sv_enc, sv_dec, model)
    
    print(f"SUCCESS: Model Output Value = {value:.4f}")
    print(f"SUCCESS: Model Output Policy Length = {len(policy)} (Expected: {len(actions)})")

if __name__ == "__main__":
    test_model()
