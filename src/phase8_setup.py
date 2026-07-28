import os

# 1. Update 02_EXPERIMENT_TRACKER.md
with open("02_EXPERIMENT_TRACKER.md", "a", encoding='utf-8') as f:
    f.write("\n\n| `008` | 2026-07-28 12:43 | Phase 8: Self-Play RL & Repaired Arena | N/A | 250 Self-Play games via AlphaZero loop | [ACTIVE LOCK] |\n\n")

# 2. Fix Greedy Agent
greedy_code = """import numpy as np

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
"""
with open("greedy_agent.py", "w", encoding='utf-8') as f:
    f.write(greedy_code)

print("Phase 8 Setup Complete.")
