import os

env_path = 'src/env.py'
with open(env_path, 'r', encoding='utf-8') as f:
    env_content = f.read()

env_content = env_content.replace(
    'os.path.join(os.path.dirname(__file__), "data"',
    'os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"'
)

# Fix action type casting in env.py (the error said "select_list is not list[int]")
# "select_list.append(action)" -> "select_list.append(int(action))"
env_content = env_content.replace(
    "select_list.append(action)",
    "select_list.append(int(action))"
)

with open(env_path, 'w', encoding='utf-8') as f:
    f.write(env_content)

bc_train_path = 'src/bc_train.py'
with open(bc_train_path, 'r', encoding='utf-8') as f:
    bc_content = f.read()

bc_content = bc_content.replace('"data/kaggle_replays"', 'os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kaggle_replays")')
# Ensure model checkpoint saves to root checkpoints
bc_content = bc_content.replace('"checkpoints/latest_model.pt"', 'os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "latest_model.pt")')
with open(bc_train_path, 'w', encoding='utf-8') as f:
    f.write(bc_content)

ppo_train_path = 'src/ppo_train.py'
with open(ppo_train_path, 'r', encoding='utf-8') as f:
    ppo_content = f.read()
    
ppo_content = ppo_content.replace('"checkpoints/latest_model.pt"', 'os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "latest_model.pt")')
ppo_content = ppo_content.replace('"ppo_results.txt"', 'os.path.join(os.path.dirname(os.path.dirname(__file__)), "ppo_results.txt")')

with open(ppo_train_path, 'w', encoding='utf-8') as f:
    f.write(ppo_content)

eval_strict_path = 'src/eval_strict.py'
with open(eval_strict_path, 'r', encoding='utf-8') as f:
    eval_content = f.read()
    
eval_content = eval_content.replace('"checkpoints/latest_model.pt"', 'os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "latest_model.pt")')

with open(eval_strict_path, 'w', encoding='utf-8') as f:
    f.write(eval_content)
    
print("Paths fixed.")
