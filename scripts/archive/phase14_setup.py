import os
import glob
import shutil

# 1. Directories
for d in ['src', 'checkpoints', 'data', 'scripts']:
    os.makedirs(d, exist_ok=True)
    
# 2. Move Python Files
py_files = glob.glob('*.py')
for f in py_files:
    if f != 'phase14_setup.py':
        dest = os.path.join('src', f)
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(f, dest)

# 3. Patch PPO GAE division-by-zero bug
ppo_train_path = os.path.join('src', 'ppo_train.py')
if os.path.exists(ppo_train_path):
    with open(ppo_train_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace advantage normalization
    content = content.replace(
        "b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)",
        "if len(b_advantages) > 1:\n                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)"
    )

    # Replace type casting for action in PPO train
    content = content.replace(
        "action = valid_actions[action_idx.item()]",
        "action = int(valid_actions[action_idx.item()])"
    )

    with open(ppo_train_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
# 4. Delete NaN weights
for pt in glob.glob('checkpoints/*.pt'):
    os.remove(pt)
if os.path.exists('best_model.pt'):
    os.remove('best_model.pt')
    
# 5. Tracker Update
with open("02_EXPERIMENT_TRACKER.md", "a", encoding="utf-8") as f:
    f.write("\n| `014` | 2026-07-28 13:25 | Phase 14: Workspace Cleanup & PPO Stabilization | N/A | Cleaned workspace into src/, patched NaN bug & type casts | [ACTIVE LOCK] |\n")

print("Setup Complete.")
