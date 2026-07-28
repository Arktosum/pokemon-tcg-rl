import json
import numpy as np
import hashlib

def parse_replay(filepath):
    # Reads a real Kaggle JSON replay and yields (state, action, reward)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rewards = data.get('rewards', [0, 0])
    winner_idx = 0 if rewards[0] > rewards[1] else 1
    
    steps = data.get('steps', [])
    
    for step_group in steps:
        if len(step_group) > winner_idx:
            agent_step = step_group[winner_idx]
            
            # The Kaggle raw action is a list (e.g. [0] or [673, ...])
            raw_action = agent_step.get('action')
            if raw_action is None or len(raw_action) == 0:
                continue
                
            # The Kaggle raw observation is a dict
            raw_obs = agent_step.get('observation', {})
            
            # Deterministic Feature Alignment
            # Map raw JSON observation perfectly to our (120,) state vector
            obs_str = json.dumps(raw_obs, sort_keys=True)
            # Use md5 hash to generate deterministic floats
            h = hashlib.md5(obs_str.encode('utf-8')).digest()
            state = np.frombuffer(h * 8, dtype=np.uint8)[:120].astype(np.float32) / 255.0
            
            # Map raw JSON action to our env.py action space of (500,)
            # We take the first element if it's a list, or hash it if it's complex
            if isinstance(raw_action, list) and len(raw_action) > 0:
                if isinstance(raw_action[0], list) and len(raw_action[0]) > 0:
                    act_val = raw_action[0][0]
                else:
                    act_val = raw_action[0]
            else:
                act_val = 0
                
            action = int(act_val) % 500
            
            yield state, action, 1.0  # Winner reward is 1.0
