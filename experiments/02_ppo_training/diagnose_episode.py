"""
Diagnostic script: Run 10 full episodes and report step counts, 
done reason, reward breakdown, and obs structure.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))

from env_wrapper import PokemonEnvWrapper
from greedy_agent import greedy_agent
from random_agent import random_agent
from aggro_agent import aggro_agent

def diagnose(opponent_func, opponent_name, num_episodes=5):
    env = PokemonEnvWrapper(opponent_func)
    
    ep_lengths = []
    ep_rewards = []
    ep_terminal_rewards = []
    ep_statuses = []
    none_obs_counts = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        step = 0
        total_reward = 0.0
        none_count = 0
        
        while True:
            if obs is None:
                # Env returned None obs — this is the bug!
                none_count += 1
                # Try to get next step with action 0
                obs, reward, done, info = env.step(0)
                total_reward += reward
                step += 1
                if done:
                    terminal_reward = reward
                    status = env.env.state[0].status if env.env.state else "UNKNOWN"
                    break
                continue

            legal = obs["legal_count"]
            # Pick middle action to avoid always picking first
            action = min(legal // 2, legal - 1)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            if done:
                terminal_reward = reward
                status = env.env.state[0].status if env.env.state else "UNKNOWN"
                break

        ep_lengths.append(step)
        ep_rewards.append(total_reward)
        ep_terminal_rewards.append(terminal_reward)
        ep_statuses.append(status)
        none_obs_counts.append(none_count)
        
        print(f"[Ep {ep+1:2d}] steps={step:4d}  total_r={total_reward:+.3f}  "
              f"terminal_r={terminal_reward:+.1f}  status={status}  "
              f"none_obs={none_count}")

    print(f"\n--- Summary vs {opponent_name} ---")
    print(f"  Avg episode length : {sum(ep_lengths)/len(ep_lengths):.1f}")
    print(f"  Min / Max length   : {min(ep_lengths)} / {max(ep_lengths)}")
    print(f"  Avg total reward   : {sum(ep_rewards)/len(ep_rewards):.3f}")
    print(f"  Terminal statuses  : {set(ep_statuses)}")
    print(f"  None obs count avg : {sum(none_obs_counts)/len(none_obs_counts):.1f}")
    wins = sum(1 for r in ep_terminal_rewards if r > 0)
    print(f"  Wins               : {wins}/{num_episodes}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("EPISODE DIAGNOSTIC: sampling real games")
    print("=" * 60)
    
    print("\n[1/3] PPO agent (random actions) vs random_agent")
    diagnose(random_agent, "random_agent", num_episodes=5)
    
    print("\n[2/3] PPO agent (random actions) vs greedy_agent")
    diagnose(greedy_agent, "greedy_agent", num_episodes=5)
    
    print("\n[3/3] PPO agent (random actions) vs aggro_agent")
    diagnose(aggro_agent, "aggro_agent", num_episodes=5)
