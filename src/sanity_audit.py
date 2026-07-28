import os
import sys
import numpy as np

def run_sanity_audit():
    # Clear poisoned weights
    if os.path.exists("checkpoints/latest_model.pt"):
        os.remove("checkpoints/latest_model.pt")
    if os.path.exists("best_model.pt"):
        os.remove("best_model.pt")
    
    from env import PTCGEnv
    env = PTCGEnv()
    obs, _ = env.reset()
    
    # 1. Assert max scalar feature <= 1.0
    state_vec = obs["obs"]
    # We know the first 60 slots are a mix of categorical / discrete, but the specific
    # stats (HP, Damage) we patched are in the continuous slots. Let's strictly check 
    # the ones we just normalized (from idx 4 to 120, skipping the Card IDs).
    # Specifically, indices 13 (id), 17 (id), 21 (id)... 
    # Just grab all scalars from idx 4 onwards, masking out values > 2 (which would only be IDs)
    scalars = [v for i, v in enumerate(state_vec[4:]) if v <= 1.0]
    max_scalar = max(scalars) if scalars else 0.0
    
    # Actually, we can check all values, but Card IDs will be > 1. 
    # Let's extract exactly the normalized ones to be perfectly sure.
    norm_vals = []
    idx = 4
    for p_idx in [0, 1]:
        norm_vals.extend(state_vec[idx:idx+9])
        idx += 9
        if state_vec[idx] > 0: # Active ID
            norm_vals.extend(state_vec[idx+1:idx+4])
        idx += 4
        for b_i in range(5):
            if state_vec[idx] > 0: # Bench ID
                norm_vals.extend(state_vec[idx+1:idx+4])
            idx += 4
            
    assert max(norm_vals) <= 1.0, f"Normalization failed! Max: {max(norm_vals)}"
    
    # 2. Assert forced loss == -1.0
    # Simulate a forced engine DQ to check reward
    env.is_done = True
    env.winner = 1
    env.current_obs.current.yourIndex = 0
    _, reward, _, _, _ = env.step(0)
    assert reward == -1.0, f"Reward logic failed! Expected -1.0, got {reward}"
    
    print(f"Sanity Audit Passed. Max scalar value is now {max(norm_vals)}.")
    with open("sanity_results.txt", "w") as f:
        f.write(f"Passed,{max(norm_vals)}")

if __name__ == "__main__":
    run_sanity_audit()
