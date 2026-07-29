import numpy as np
import random
import sys
import os

# Add Kaggle simulation engine to path to import OptionType if needed
ENGINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_submission", "sample_submission")
if ENGINE_PATH not in sys.path:
    sys.path.append(ENGINE_PATH)

try:
    from cg.api import OptionType
except ImportError:
    pass

class RandomAgent:
    def __init__(self):
        pass
        
    def act(self, obs, env=None):
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            return 0
        return int(np.random.choice(valid_actions))

class AdvancedHeuristicAgent:
    def __init__(self):
        pass
        
    def act(self, obs, env):
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) == 0:
            return 0
            
        if env is None or env.current_obs is None or env.current_obs.select is None:
            # Fallback to random if no env semantic info is provided
            return int(np.random.choice(valid_actions))
            
        options = env.current_obs.select.option
        
        # We want to prioritize based on OptionType
        # 13: ATTACK
        # 9: EVOLVE
        # 7: PLAY (Cards/Supporters)
        # 8: ATTACH (Energy/Tools)
        # 10: ABILITY
        # 14: END
        
        # Categorize valid actions
        attacks = []
        evolves = []
        plays = []
        attaches = []
        abilities = []
        others = []
        ends = []
        
        for a in valid_actions:
            a = int(a)
            if a < len(options):
                opt = options[a]
                # If OptionType is available we can check opt.type, else we just guess by type integer
                # Assuming enum values: ATTACK=13, EVOLVE=9, PLAY=7, ATTACH=8, ABILITY=10, END=14
                t = opt.type
                # Handle Enum if it comes as an enum object
                t_val = t.value if hasattr(t, "value") else t
                
                if t_val == 13:
                    attacks.append(a)
                elif t_val == 9:
                    evolves.append(a)
                elif t_val == 7:
                    plays.append(a)
                elif t_val == 8:
                    attaches.append(a)
                elif t_val == 10:
                    abilities.append(a)
                elif t_val == 14:
                    ends.append(a)
                else:
                    others.append(a)
            else:
                others.append(a)
                
        # Priority 1: Attacks
        if len(attacks) > 0:
            return random.choice(attacks)
            
        # Priority 2: Evolve
        if len(evolves) > 0:
            return random.choice(evolves)
            
        # Priority 3: Play Cards (Supporters/Items)
        if len(plays) > 0:
            return random.choice(plays)
            
        # Priority 4: Attach Energy
        if len(attaches) > 0:
            return random.choice(attaches)
            
        # Priority 5: Abilities
        if len(abilities) > 0:
            return random.choice(abilities)
            
        # Priority 6: Others (e.g. YES/NO prompts, retreat)
        if len(others) > 0:
            return random.choice(others)
            
        # Priority 7: End Turn
        if len(ends) > 0:
            return random.choice(ends)
            
        return int(random.choice(valid_actions))
