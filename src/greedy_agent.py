import numpy as np

class GreedyAgent:
    def __init__(self):
        pass
        
    def act(self, obs):
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) == 0:
            return 0
            
        # Repaired Logic: The previous logic picked the maximum index, which likely mapped 
        # to "Concede" or "Draw" leading to deck-out.
        # Without semantic mapping, a safe "greedy" approach is to prioritize lower-index actions
        # (which typically map to attacks and energy attachments) while avoiding Action 0 (Pass).
        best_action = 0
        for a in valid_actions:
            if a != 0:
                best_action = int(a)
                break
                
        if best_action == 0 and len(valid_actions) > 0:
            best_action = int(valid_actions[0])
            
        return best_action
