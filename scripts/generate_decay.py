import os

with open('scripts/train_fictitious_ppo.py', 'r') as f:
    content = f.read()

# 1. 100% Greedy Opponent
content = content.replace(
'''        # Opponent Selection
        rand = random.random()
        if rand < 0.40:
            opp_name = "Greedy"
            opp_agent = greedy_agent
        elif rand < 0.70:
            opp_name = "FrozenBC"
            opp_agent = frozen_ref
        elif rand < 0.90:
            opp_name = "Advanced"
            opp_agent = adv_agent
        else:
            opp_name = "Random"
            opp_agent = random_agent''',
'''        # Opponent Selection
        opp_name = "Greedy"
        opp_agent = greedy_agent'''
)

# 2. Linear KL Decay
content = content.replace(
'''    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()''',
'''    for episode in range(1, num_episodes + 1):
        current_kl_coef = 0.02 * (1.0 - (episode - 1) / num_episodes)
        obs, _ = env.reset()'''
)

# 3. Replace KL coef usage
content = content.replace(
'''loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy + kl_coef * kl_div''',
'''loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy + current_kl_coef * kl_div'''
)

# 4. Print statement modifications
content = content.replace(
'''print(f"Ep {episode:03d} vs {opp_name:10s} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f} | KL: {last_kl:.6f} | Steps: {step}")''',
'''print(f"Ep {episode:03d} vs {opp_name:10s} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f} | KL: {last_kl:.6f} | KL_coef: {current_kl_coef:.4f} | Steps: {step}")'''
)
content = content.replace(
'''print(f"Ep {episode:05d} vs {opp_name:10s} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f} | KL: {last_kl:.6f}", flush=True)''',
'''print(f"Ep {episode:05d} vs {opp_name:10s} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f} | KL: {last_kl:.6f} | KL_coef: {current_kl_coef:.4f}", flush=True)'''
)

# 5. Checkpoint paths
content = content.replace('TITAN_FICTITIOUS_PPO_01_epoch_', 'TITAN_KL_DECAY_01_epoch_')
content = content.replace('TITAN_FICTITIOUS_PPO_FINAL.pt', 'TITAN_KL_DECAY_FINAL.pt')
content = content.replace('episode % 500 == 0', 'episode % 100 == 0')

with open('scripts/train_kl_decay.py', 'w') as f:
    f.write(content)
