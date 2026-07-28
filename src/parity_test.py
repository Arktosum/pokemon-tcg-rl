import json
import os
import torch
import numpy as np

def run_parity_test():
    replay_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")
    
    passed = 0
    failed = 0
    
    if not os.path.exists(replay_dir):
        print("Fail: No replay dir found")
        return
        
    for filename in os.listdir(replay_dir):
        if filename.endswith(".json"):
            with open(os.path.join(replay_dir, filename), 'r') as f:
                data = json.load(f)
                
                for step in data.get('steps', []):
                    # In our synthetic replays, policy is a probability distribution over valid actions.
                    # We can check if any action with policy > 0 has mask == 1.
                    policy = np.array(step['policy'])
                    mask = np.array(step['mask'])
                    
                    # Find actions the "expert" took (probability > 0)
                    expert_actions = np.where(policy > 0)[0]
                    
                    for ea in expert_actions:
                        if mask[ea] != 1:
                            failed += 1
                        else:
                            passed += 1
                            
    print(f"Parity Audit Complete: {passed} Valid Actions, {failed} Illegal Actions.")
    if failed == 0 and passed > 0:
        print("PARITY: PASS")
    else:
        print("PARITY: FAIL")

if __name__ == "__main__":
    run_parity_test()
