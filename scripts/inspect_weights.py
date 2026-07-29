import torch
import os

ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TOP_ELO_PPO_PEAK.pt")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
state_dict = checkpoint.get("model_state_dict", checkpoint)

print(f"Loaded weights from {ckpt_path}")
for k, v in state_dict.items():
    if "transformer.layers" in k or "embedding" in k or "fc" in k:
        print(f"{k}: {v.shape}")
