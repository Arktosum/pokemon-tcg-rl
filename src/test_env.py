import env
import numpy as np

def main():
    print("Testing PTCG Environment Wrapper...")
    game_env = env.PTCGEnv()
    obs, info = game_env.reset()
    
    print(f"Initial Obs Vector Shape: {obs['obs'].shape}")
    print(f"Action Mask Shape: {obs['action_mask'].shape}")
    
    steps = 0
    done = False
    reward = 0
    
    while not done and steps < 1000:
        # Random valid action
        mask = obs['action_mask']
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 0
            
        obs, reward, done, truncated, info = game_env.step(int(action))
        steps += 1
        
    print(f"Game finished in {steps} steps.")
    print(f"Final Reward: {reward}")
    print(f"Final State Vector: {obs['obs'][:10]}")
    game_env.close()

if __name__ == '__main__':
    main()
