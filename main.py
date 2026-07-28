import os
import random

# Fallback setup
MODEL_LOADED = False
ort_sess = None
np = None
csv = None

try:
    import sys
    import csv as local_csv
    import numpy as real_np
    
    csv = local_csv
    np = real_np
    
    try:
        KAGGLE_AGENT_PATH = os.path.dirname(__file__)
    except NameError:
        KAGGLE_AGENT_PATH = '/kaggle_simulations/agent/'

    # Inject ONNX runtime
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, 'src', 'kaggle_libs', 'onnxruntime_pkg'))

    import onnxruntime as ort
    model_path = os.path.join(KAGGLE_AGENT_PATH, 'titan_model.onnx')
    if os.path.exists(model_path):
        ort_sess = ort.InferenceSession(model_path)
        MODEL_LOADED = True
    else:
        print(f"Error: ONNX model not found at {model_path}")

except Exception as e:
    print(f"Global Scope Initialization Error: {e}")
    # Fallback completely enabled

def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        try:
            file_path = os.path.join(os.path.dirname(__file__), "deck.csv")
        except NameError:
            file_path = "/kaggle_simulations/agent/deck.csv"
            
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"

    try:
        with open(file_path, "r") as file:
            csv_lines = file.read().split("\n")
        deck = []
        for i in range(60):
            deck.append(int(csv_lines[i]))
        return deck
    except Exception as e:
        print(f"Deck read error: {e}")
        # Return fallback 60 integers just in case (e.g. 1158)
        return [1158] * 60

def agent(obs_dict, conf_dict=None) -> list[int]:
    try:
        # Check initial step
        if isinstance(obs_dict, dict):
            select = obs_dict.get("select")
        else:
            select = getattr(obs_dict, "select", None)
            
        if select is None:
            return read_deck_csv()
            
        # Parse maxCount and option
        if isinstance(select, dict):
            max_count = select.get("maxCount", 1)
            options = select.get("option", [])
        else:
            max_count = getattr(select, "maxCount", 1)
            options = getattr(select, "option", [])
            
        num_options = len(options) if options else 1
        max_count = min(max_count, num_options)
        
        # Phase 42: ONNX Top-K Inference
        if not MODEL_LOADED or ort_sess is None or np is None:
            return random.sample(list(range(num_options)), max_count)
            
        # ONNX state inference
        state_array = np.zeros((1, 120), dtype=np.float32)
        outputs = ort_sess.run(None, {'state': state_array})
        policy_probs = outputs[0][0] # Assuming shape is (1, action_dim)
        
        # Mask valid options
        # Sort options based on the logits, highest first
        valid_indices = list(range(num_options))
        # policy_probs might exceed valid_indices if action_dim > num_options
        # but we only sort the valid_indices
        sorted_actions = sorted(valid_indices, key=lambda idx: policy_probs[idx] if idx < len(policy_probs) else -1e9, reverse=True)
        
        return sorted_actions[:max_count]
        
    except Exception as e:
        print(f"CRITICAL AGENT ERROR: {e}")
        
        # Ultimate fallback
        if isinstance(obs_dict, dict):
            select = obs_dict.get("select")
        else:
            select = getattr(obs_dict, "select", None)
            
        if select is None:
            return [1158] * 60
            
        if isinstance(select, dict):
            max_count = select.get("maxCount", 1)
            options = select.get("option", [])
        else:
            max_count = getattr(select, "maxCount", 1)
            options = getattr(select, "option", [])
            
        num_options = len(options) if options else 1
        return random.sample(list(range(num_options)), min(max_count, num_options))
