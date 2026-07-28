import os
import sys
import time
import torch
import numpy as np
from model import PokemonAlphaNet
from puct import PUCTSearch

model = None

def agent(obs, config):
    global model
    start_time = time.perf_counter()
    
    if model is None:
        model = PokemonAlphaNet()
        # Devil's Advocate Check: Use os.path.dirname(__file__) for robust pathing on Kaggle
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # Time budget limit: Kaggle strict timeout usually 1-3 seconds. We cap at 2.0s
    time_limit = config.get('actTimeout', 2.0) if hasattr(config, 'get') else 2.0
    
    # In a real environment, we'd map obs to our state tensor here.
    # Simulated search:
    # searcher = PUCTSearch(model, num_simulations=25)
    # 
    # Devil's advocate timeout logic:
    # if (time.perf_counter() - start_time) > (time_limit - 0.5):
    #     force_early_stop()
    
    return 0 
