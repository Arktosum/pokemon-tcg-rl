import sys
import os

# Append the PARENT directory of 'cg' so 'from cg.api import ...' works
parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'

if not os.path.exists(parent_cg_path):
    raise FileNotFoundError(f"Could not find parent cg directory at {parent_cg_path}")

print(f"Appended to sys.path: {parent_cg_path}")
sys.path.append(parent_cg_path)

try:
    from cg.api import all_card_data, all_attack, to_observation_class
    print("SUCCESS: Imported cg.api successfully!")
    cards = all_card_data()
    print(f"Loaded {len(cards)} cards.")
    attacks = all_attack()
    print(f"Loaded {len(attacks)} attacks.")
except Exception as e:
    print(f"FAILED to load cg.api: {e}")
