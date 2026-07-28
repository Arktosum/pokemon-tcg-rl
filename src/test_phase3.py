import time
import numpy as np
import torch
from env import PTCGEnv
from model import PokemonAlphaNet
from puct import PUCTSearch

def run_test():
    print("--- Phase 3: Dual-Head NN & PUCT Validation ---")
    
    # 1. Instantiate Environment
    env = PTCGEnv()
    obs, info = env.reset()
    state_vec = obs["obs"]
    action_mask = obs["action_mask"]
    
    print(f"Environment initialized.")
    print(f"State vector shape: {state_vec.shape}")
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Valid actions count: {np.sum(action_mask)}")
    
    # 2. Instantiate Model
    model = PokemonAlphaNet()
    model.eval()
    
    # 3. Shape Verification
    print("\n--- Verifying Network Shapes ---")
    state_tensor = torch.tensor(state_vec, dtype=torch.float32)
    mask_tensor = torch.tensor(action_mask, dtype=torch.int8)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        policy, value = model(state_tensor, mask_tensor)
    inf_latency = (time.perf_counter() - start_time) * 1000
    
    print(f"Policy shape: {policy.shape} (Expected: (1, 500))")
    print(f"Value shape: {value.shape} (Expected: (1, 1))")
    print(f"Single-step inference latency: {inf_latency:.2f} ms")
    
    assert policy.shape == (1, 500), "Policy shape mismatch!"
    assert value.shape == (1, 1), "Value shape mismatch!"
    
    # Verify masking logic (illegal actions should have prob 0)
    policy_np = policy.squeeze(0).numpy()
    illegal_sum = np.sum(policy_np[action_mask == 0])
    print(f"Sum of probabilities for illegal actions: {illegal_sum}")
    assert illegal_sum < 1e-5, "Action masking failed!"
    
    # 4. PUCT 10-Simulation Rollout
    print("\n--- Running PUCT Search (10 simulations) ---")
    searcher = PUCTSearch(model, num_simulations=10)
    
    start_search = time.perf_counter()
    action_probs = searcher.search(state_vec, action_mask, to_play=0)
    search_latency = (time.perf_counter() - start_search) * 1000
    
    print(f"PUCT Search latency: {search_latency:.2f} ms")
    print("\nResulting Action Probabilities (Non-zero):")
    for a, p in enumerate(action_probs):
        if p > 0:
            print(f"Action {a}: {p:.4f}")
            
    print("\nValidation Complete. All systems nominal.")

if __name__ == "__main__":
    run_test()
