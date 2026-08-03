from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

@dataclass
class ParsedObs:
    enc_index: List[int]
    enc_value: List[float]
    enc_offset: List[int]
    dec_index: List[int]
    dec_value: List[float]
    dec_offset: List[int]
    num_options: int
    max_count: int
    prizes_left: int = 6

@dataclass
class StepResult:
    obs: Optional[ParsedObs]
    reward: float
    done: bool
    info: dict

@dataclass
class RolloutBatch:
    # B = batch size (steps in rollout)
    enc_indices: torch.Tensor       # [B, max_enc_len]
    enc_values: torch.Tensor        # [B, max_enc_len]
    enc_offsets: torch.Tensor       # [B, 24]
    
    # We flatten dec_inputs or use a list of lists depending on the network expectations.
    # The network expects decoder_inputs as a list (batch) of list (options) of tuples (idxs, vals, offset).
    # Since we can't easily tensorize a jagged 3D structure with different lengths per option,
    # RolloutBatch keeps them as the raw format the model accepts, or we pad them.
    # The TitanTransformer model forward expects: decoder_inputs: list[list[tuple]]
    decoder_inputs: List[List[Tuple[torch.Tensor, torch.Tensor, int]]]
    
    action_masks: torch.Tensor      # [B, max_options] (boolean mask where True = ignore/invalid, False = valid)
    
    actions: torch.Tensor           # [B, max_count] - The actions actually taken
    rewards: torch.Tensor           # [B]
    dones: torch.Tensor             # [B]
    old_log_probs: torch.Tensor     # [B]
    values: torch.Tensor            # [B]
