import os

with open('scripts/train_fictitious_ppo.py', 'r') as f:
    content = f.read()

# 1. 100% Greedy Opponent
orig_opp = """        # Opponent Selection
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
            opp_agent = random_agent"""

new_opp = """        # Opponent Selection
        opp_name = "Greedy"
        opp_agent = greedy_agent"""

content = content.replace(orig_opp, new_opp)

# 2. Entropy coef
content = content.replace('entropy_coef = 0.05', 'entropy_coef = 0.01')

# 3. Checkpoint paths
content = content.replace('TITAN_FICTITIOUS_PPO_01_epoch_', 'TITAN_LOW_ENTROPY_01_epoch_')
content = content.replace('TITAN_FICTITIOUS_PPO_FINAL.pt', 'TITAN_LOW_ENTROPY_FINAL.pt')
content = content.replace('episode % 500 == 0', 'episode % 100 == 0')

with open('scripts/train_low_entropy.py', 'w') as f:
    f.write(content)
