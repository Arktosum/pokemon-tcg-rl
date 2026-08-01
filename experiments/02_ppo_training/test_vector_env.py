import sys
import os
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))
from random_agent import random_agent
from vector_env import VectorEnv
from train_ppo import collate_observations

def test_vector_env_creation():
    env = VectorEnv(random_agent, num_envs=2)
    obs = env.reset()
    assert len(obs) == 2
    assert "enc_indices" in obs[0]
    
    actions = [0, 0]
    next_obs, rewards, dones, infos = env.step(actions)
    
    assert len(next_obs) == 2
    assert len(rewards) == 2
    assert len(dones) == 2
    assert len(infos) == 2
    
    env.close()

import numpy as np

def test_batched_collate():
    obs1 = {
        "enc_indices": np.array([1, 2, 3], dtype=np.int32),
        "enc_offsets": np.array([0, 1], dtype=np.int32),
        "enc_weights": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "dec_indices": np.array([4, 5], dtype=np.int32),
        "dec_offsets": np.array([0], dtype=np.int32),
        "dec_weights": np.array([1.0, 1.0], dtype=np.float32),
        "legal_count": 2
    }
    obs2 = {
        "enc_indices": np.array([4, 5], dtype=np.int32),
        "enc_offsets": np.array([0], dtype=np.int32),
        "enc_weights": np.array([1.0, 1.0], dtype=np.float32),
        "dec_indices": np.array([6, 7, 8], dtype=np.int32),
        "dec_offsets": np.array([0, 1], dtype=np.int32),
        "dec_weights": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "legal_count": 3
    }
    
    col = collate_observations([obs1, obs2])
    
    assert col["enc_indices"].tolist() == [1, 2, 3, 4, 5]
    assert col["enc_offsets"].tolist() == [0, 1, 3]  # The second sample's offset starts at 3 because sample 1 has 3 indices
    assert col["dec_offsets"].tolist() == [0, 2, 3]  # Sample 1 has 2 indices, so sample 2's offsets are +2
    assert col["legal_counts"].tolist() == [2, 3]

def test_batched_collate_empty():
    obs_empty = {
        "enc_indices": torch.tensor([], dtype=torch.int32),
        "enc_offsets": torch.tensor([], dtype=torch.int32),
        "enc_weights": torch.tensor([], dtype=torch.float32),
        "dec_indices": torch.tensor([], dtype=torch.int32),
        "dec_offsets": torch.tensor([], dtype=torch.int32),
        "dec_weights": torch.tensor([], dtype=torch.float32),
        "legal_count": 0
    }
    col = collate_observations([obs_empty, obs_empty])
    assert col["enc_indices"].shape[0] == 0
    assert col["dec_offsets"].tolist() == []
