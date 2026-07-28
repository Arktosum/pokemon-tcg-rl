import torch
import torch.nn.functional as F
from model import PokemonActorCritic
from replay_parser import parse_replays

dataset_dict = parse_replays()
states = dataset_dict['state'][:32]
masks = dataset_dict['mask'][:32]
zs = dataset_dict['z'][:32]

model = PokemonActorCritic()
pred_policy, pred_value = model(states, masks)
print(f"pred_value shape: {pred_value.shape}")
print(f"zs shape: {zs.shape}")
print(f"v_loss: {F.mse_loss(pred_value, zs).item()}")
