import yaml
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class TitanConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    max_actions: int
    n_words: int

@dataclass
class PPOConfig:
    total_episodes: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    save_freq: int
    max_steps: int

@dataclass
class CurriculumConfig:
    checkpoint_freq: int
    validation_freq: int
    sampling_weights: Dict[str, float]
    rule_based_bots: List[str]

def load_config(yaml_path: str = "config.yaml"):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, yaml_path)
    with open(full_path, 'r') as f:
        data = yaml.safe_load(f)
    
    titan_cfg = TitanConfig(**data['model'])
    ppo_cfg = PPOConfig(**data['ppo'])
    curr_cfg = CurriculumConfig(**data['curriculum']) if 'curriculum' in data else None
    return titan_cfg, ppo_cfg, curr_cfg
