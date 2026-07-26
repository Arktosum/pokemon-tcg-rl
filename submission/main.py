import os
import sys
import torch
from cg.api import Observation, to_observation_class

# Add current directory to path for local module imports on Kaggle
sys.path.append(os.path.dirname(__file__))
from transformer_policy import MyModel, get_encoder_input, get_decoder_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
deck_list = None

def load_model():
    global model, deck_list
    if model is None:
        model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1).to(device)
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        deck_list = read_deck_csv()

def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().strip().split("\n")
    deck = [int(x) for x in csv[:60]]
    return deck

def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()
        
    load_model()
    
    sv_enc = get_encoder_input(obs, deck_list)
    legal_actions = [[i] for i in range(len(obs.select.option))]
    sv_dec = get_decoder_input(obs, legal_actions)
    
    enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32, device=device)
    enc_val = torch.tensor(sv_enc.value, dtype=torch.float32, device=device)
    enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32, device=device)
    dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32, device=device)
    dec_val = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
    dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
    
    with torch.no_grad():
        _, logits = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
        action_idx = torch.argmax(logits).item()
        
    return legal_actions[action_idx]
