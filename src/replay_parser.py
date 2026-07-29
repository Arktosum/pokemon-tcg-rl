import json
import sys
import os

# Add Kaggle engine to path for cg.api
SRC_PATH = os.path.dirname(__file__)
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

ENGINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_submission", "sample_submission")
if ENGINE_PATH not in sys.path:
    sys.path.append(ENGINE_PATH)

from env import PTCGEnv
from cg.api import to_observation_class

_env_instance = PTCGEnv()

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
            raw_obs = agent_step.get('observation')
            if raw_obs is None or raw_obs.get('current') is None:
                continue
            
            # Get valid hand-crafted feature vector
            try:
                state_dict = _env_instance._process_obs(raw_obs)
                state = state_dict['obs']
            except Exception as e:
                # If the env fails to process this observation, drop the step
                continue
            
            # Extract action safely
            if isinstance(raw_action, list):
                if isinstance(raw_action[0], list):
                    act_val = raw_action[0][0] if len(raw_action[0]) > 0 else 0
                else:
                    act_val = raw_action[0]
            else:
                act_val = raw_action
                
            action = int(act_val)
            
            # Reject out-of-bounds actions rather than wrapping them silently
            if action < 0 or action >= 500:
                continue
            
            yield state, action, 1.0  # Winner reward is 1.0
