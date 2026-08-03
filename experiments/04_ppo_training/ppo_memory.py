import torch
import torch
import numpy as np
from typing import List, Tuple, Any
from ppo_types import ParsedObs, StepResult, RolloutBatch

class RolloutBuffer:
    def __init__(self, max_steps: int, max_seq_len: int, max_options: int, device: str = "cpu"):
        self.device = device
        self.max_steps = max_steps
        self.max_seq_len = max_seq_len
        self.max_options = max_options
        self.ptr = 0

        # Pre-allocate contiguous memory blocks
        self.enc_indices = torch.zeros((max_steps, max_seq_len), dtype=torch.long, device=device)
        self.enc_values = torch.zeros((max_steps, max_seq_len), dtype=torch.float32, device=device)
        self.enc_offsets = torch.zeros((max_steps, 24), dtype=torch.long, device=device)
        
        self.decoder_inputs = [None] * max_steps
        
        self.action_masks = torch.ones((max_steps, max_options), dtype=torch.bool, device=device)
        self.actions = torch.full((max_steps, max_options), -1, dtype=torch.long, device=device)
        self.rewards = torch.zeros((max_steps,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_steps,), dtype=torch.bool, device=device)
        self.values = torch.zeros((max_steps,), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((max_steps,), dtype=torch.float32, device=device)

    def clear(self):
        self.ptr = 0

    def add(self, parsed_obs: ParsedObs, action: List[int], reward: float, done: bool, value: float, log_prob: float):
        if self.ptr >= self.max_steps:
            raise RuntimeError("RolloutBuffer is full!")
            
        # Insert directly into pre-allocated memory
        enc_len = min(self.max_seq_len, len(parsed_obs.enc_index))
        
        self.enc_indices[self.ptr, :enc_len] = torch.as_tensor(np.array(parsed_obs.enc_index[:enc_len], dtype=np.int64), device=self.device)
        if enc_len < self.max_seq_len:
             self.enc_indices[self.ptr, enc_len:] = 0
             
        self.enc_values[self.ptr, :enc_len] = torch.as_tensor(np.array(parsed_obs.enc_value[:enc_len], dtype=np.float32), device=self.device)
        if enc_len < self.max_seq_len:
             self.enc_values[self.ptr, enc_len:] = 0.0

        off_len = min(24, len(parsed_obs.enc_offset))
        self.enc_offsets[self.ptr, :off_len] = torch.as_tensor(np.array(parsed_obs.enc_offset[:off_len], dtype=np.int64), device=self.device)
        if off_len < 24:
             self.enc_offsets[self.ptr, off_len:] = 0
        
        # Build decoder_inputs
        num_options = min(self.max_options, parsed_obs.num_options)
        dec_inputs_list = []
        for a in range(num_options):
            start = parsed_obs.dec_offset[a] if a < len(parsed_obs.dec_offset) else len(parsed_obs.dec_index)
            end = parsed_obs.dec_offset[a+1] if a+1 < len(parsed_obs.dec_offset) else len(parsed_obs.dec_index)
            
            idxs = torch.as_tensor(np.array(parsed_obs.dec_index[start:end], dtype=np.int64), device=self.device)
            vals = torch.as_tensor(np.array(parsed_obs.dec_value[start:end], dtype=np.float32), device=self.device)
            dec_inputs_list.append((idxs, vals, start))
        
        self.decoder_inputs[self.ptr] = dec_inputs_list
        
        # Action mask: all false means valid. Assuming all generated options are valid for now.
        mask = torch.ones(self.max_options, dtype=torch.bool, device=self.device)
        mask[:num_options] = False
        self.action_masks[self.ptr] = mask

        # Pad actions list to max_count if needed
        act_len = len(action)
        self.actions[self.ptr, :act_len] = torch.as_tensor(np.array(action, dtype=np.int64), device=self.device)
        if act_len < self.max_options:
             self.actions[self.ptr, act_len:] = -1
             
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1

    def get_batch(self) -> RolloutBatch:
        if self.ptr == 0:
            raise ValueError("Buffer is empty!")

        # Actually, we can return the exact slice of the pre-allocated arrays up to self.ptr
        # However, to save memory in the batch, we can crop enc_indices to the max length *in this batch*
        # (Though returning the slice directly is faster and zero-copy)
        
        # Determine actual max length used in this batch to trim
        max_enc_len = 0
        for i in range(self.ptr):
            # Find the last non-zero index to approximate length if we want to trim, 
            # but for speed it's often easier to just use the full tensor or use a known max length.
            pass # Skipping trimming for simplicity, using the pre-allocated slice
            
        return RolloutBatch(
            enc_indices=self.enc_indices[:self.ptr],
            enc_values=self.enc_values[:self.ptr],
            enc_offsets=self.enc_offsets[:self.ptr],
            decoder_inputs=self.decoder_inputs[:self.ptr],
            action_masks=self.action_masks[:self.ptr],
            actions=self.actions[:self.ptr],
            rewards=self.rewards[:self.ptr],
            dones=self.dones[:self.ptr],
            values=self.values[:self.ptr],
            old_log_probs=self.log_probs[:self.ptr]
        )
