import time
import multiprocessing as mp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))
from vector_env import VectorEnv
from random_agent import random_agent

def test_dynamic_opponent_swapping():
    print("[TEST] Booting VectorEnv...")
    env = VectorEnv(random_agent, num_envs=2)
    
    print("[TEST] Initial Reset...")
    env.reset()
    
    print("[TEST] Stepping 10 times to dirty the envs...")
    for _ in range(10):
        env.step([[0]] * 2)  # dummy action
        
    opponents = ["random_agent", "greedy_agent", "setup_agent", "tactical_agent", "self_play_agent"]
    
    for opp in opponents:
        print(f"[TEST] Dynamic Reset -> {opp}")
        try:
            obs = env.reset(opp)
            assert len(obs) == 2, "Reset should return 2 obs"
            assert obs[0] is not None, "Obs should not be None"
            print(f"[TEST] Success with {opp}!")
        except Exception as e:
            print(f"[TEST] FAILED on {opp}: {e}")
            
    print("[TEST] Closing env...")
    env.close()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    test_dynamic_opponent_swapping()
