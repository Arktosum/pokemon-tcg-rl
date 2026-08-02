import sys
import os
import torch
import pytest

_ppo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ppo_dir not in sys.path:
    sys.path.insert(0, _ppo_dir)

from train_ppo import calc_entropy, compute_gae
from env_wrapper import PokemonPPOEnv

def test_entropy_calculation():
    logits = torch.tensor([[10.0, 10.0, -1e9]])
    mask = torch.tensor([[False, False, True]])
    
    entropy = calc_entropy(logits, mask)
    # The two valid logits are equal, so probs are [0.5, 0.5, 0]
    # entropy is -(0.5*ln(0.5) + 0.5*ln(0.5)) = ln(2) = 0.6931
    assert not torch.isnan(entropy)
    assert abs(entropy.item() - 0.6931) < 1e-3

def test_gae_calculation():
    rewards = [1.0, 1.0, -1.0]
    values = [0.5, 0.5, 0.5]
    dones = [0, 0, 1]
    next_value = 100.0 # Should be ignored because last step is done
    
    adv = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
    assert len(adv) == 3
    # last step is done -> next_val is 0.0
    # delta = -1.0 + 0.99*0 - 0.5 = -1.5
    assert adv[2] == -1.5
    
def test_env_wrapper():
    env = PokemonPPOEnv()
    obs = env.reset()
    assert env.my_prizes == 6
    assert env.op_prizes == 6
    
    # Check if observation has the correct structure or fallback
    assert obs is not None
    
    obs, reward, done, info = env.step([0])
    # Just checking it returns 4 elements and types are somewhat correct
    assert isinstance(done, (bool, int, list))
