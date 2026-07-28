import os
import re

# 1. Update 03_META_RESEARCH.md
with open("03_META_RESEARCH.md", "a", encoding="utf-8") as f:
    f.write("\n## Residual Network Backbone (Phase 17)\n")
    f.write("- **Architecture:** Replaced MLP with 4 Residual Blocks (Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> Skip Add -> ReLU).\n")
    f.write("- **Hidden Dimension:** 256.\n")
    f.write("- **Memory Management:** Aggressive gc.collect() and empty_cache() during PPO rollouts to prevent VRAM OOM.\n")

# 2. Update 02_EXPERIMENT_TRACKER.md
with open("02_EXPERIMENT_TRACKER.md", "a", encoding="utf-8") as f:
    f.write("\n| `017` | 2026-07-28 13:54 | Phase 17: Residual Network Scale-Up | N/A | 4-layer ResNet-256, 30 epoch BC, 2000 episode PPO | [ACTIVE LOCK] |\n")

# 3. Rewrite model.py
model_code = """import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        res = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return F.relu(out + res)

class PokemonActorCritic(nn.Module):
    def __init__(self, num_card_types=3000, emb_dim=32, scalar_dim=58, trunk_dim=256, num_actions=500, num_res_blocks=4):
        super().__init__()
        self.num_cats = 12
        self.card_embedding = nn.Embedding(num_card_types, emb_dim)
        
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        in_dim = self.num_cats * emb_dim + 128
        self.proj_in = nn.Linear(in_dim, trunk_dim)
        
        self.res_blocks = nn.ModuleList([ResidualBlock(trunk_dim) for _ in range(num_res_blocks)])
        
        self.policy_head = nn.Linear(trunk_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self.cat_indices = [13, 17, 21, 25, 29, 33, 46, 50, 54, 58, 62, 66]
        
    def forward(self, state, action_mask=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
                
        B = state.shape[0]
        device = state.device
        
        cat_tensors = state[:, self.cat_indices].long()
        scalar_mask = torch.ones(70, dtype=torch.bool, device=device)
        scalar_mask[self.cat_indices] = False
        scalar_tensors = state[:, :70][:, scalar_mask]
        
        emb_out = self.card_embedding(cat_tensors)
        emb_out = emb_out.view(B, -1)
        
        scalar_out = self.scalar_proj(scalar_tensors)
        x = torch.cat([emb_out, scalar_out], dim=-1)
        
        x = self.proj_in(x)
        for block in self.res_blocks:
            x = block(x)
            
        logits = self.policy_head(x)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
            
        policy_probs = F.softmax(logits, dim=-1)
        value = self.value_head(x)
        
        return policy_probs, value
"""
with open("src/model.py", "w", encoding="utf-8") as f:
    f.write(model_code)

# 4. Modify ppo_train.py
with open("src/ppo_train.py", "r", encoding="utf-8") as f:
    ppo_content = f.read()

ppo_content = ppo_content.replace("import os", "import os\nimport gc")
ppo_content = ppo_content.replace("num_episodes = 2500", "num_episodes = 2000")
ppo_content = ppo_content.replace("buffer.clear()", "buffer.clear()\n            gc.collect()\n            if torch.cuda.is_available():\n                torch.cuda.empty_cache()")

with open("src/ppo_train.py", "w", encoding="utf-8") as f:
    f.write(ppo_content)

# 5. Modify bc_train.py to save model
with open("src/bc_train.py", "r", encoding="utf-8") as f:
    bc_content = f.read()

if "torch.save" not in bc_content:
    bc_content = bc_content.replace("print(\"\\nBehavioral Cloning Complete.\")", "import os\n    os.makedirs('checkpoints', exist_ok=True)\n    torch.save({'model_state_dict': model.state_dict()}, 'checkpoints/latest_model.pt')\n    print(\"\\nBehavioral Cloning Complete.\")")
    with open("src/bc_train.py", "w", encoding="utf-8") as f:
        f.write(bc_content)

# 6. Purge weights
ckpt_path = os.path.join("checkpoints", "latest_model.pt")
if os.path.exists(ckpt_path):
    os.remove(ckpt_path)

print("Setup Complete Phase 17.")
