import sys
import os
sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from src.env.tcg_env import PokemonTCGEnv
from src.model.heuristic_bot import agent as rule_agent

def test_env():
    deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    
    env = PokemonTCGEnv(deck)
    obs = env.reset()
    
    print("Environment reset successfully.")
    
    steps = 0
    while True:
        action = rule_agent(obs)
        obs, reward, done = env.step(action)
        steps += 1
        
        if done:
            print(f"Game finished in {steps} steps.")
            break
            
        if steps >= 20:
            print("Successfully stepped environment 20 times.")
            break

if __name__ == "__main__":
    test_env()
