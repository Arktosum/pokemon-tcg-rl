import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Iterator
from ppo_types import RolloutBatch

class LossMetrics:
    def __init__(self, actor_loss: float, critic_loss: float, entropy: float, kl_divergence: float, explained_variance: float, grad_norm: float):
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.entropy = entropy
        self.kl_divergence = kl_divergence
        self.explained_variance = explained_variance
        self.grad_norm = grad_norm

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    r = rewards.cpu().numpy()
    v = values.cpu().numpy()
    d = dones.cpu().numpy()
    adv = np.zeros_like(r, dtype=np.float32)
    lastgaelam = 0.0
    
    for t in reversed(range(len(r))):
        if t == len(r) - 1:
            nextnonterminal = 1.0 - float(d[t])
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - float(d[t])
            nextvalues = v[t + 1]
            
        delta = r[t] + gamma * nextvalues * nextnonterminal - v[t]
        adv[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
    advantages = torch.as_tensor(adv, device=rewards.device, dtype=rewards.dtype).view(-1)
    returns = (advantages + values.view(-1)).view(-1)
    return advantages, returns

def generate_minibatches(batch_size: int, minibatch_size: int) -> Iterator[np.ndarray]:
    """Pure generator that yields shuffled indices for minibatches."""
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        yield indices[start:end]

def update_ppo(model: nn.Module, optimizer: torch.optim.Optimizer, batch: RolloutBatch, next_value: float, 
               clip_coef: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5,
               update_epochs: int = 4, minibatch_size: int = 256) -> LossMetrics:
    
    # 1. Compute advantages and returns (Pure function)
    advantages, returns = compute_gae(batch.rewards, batch.values, batch.dones, next_value)
    
    # Standardize advantages across the entire batch
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    batch_size = len(batch.rewards)
    
    if minibatch_size > batch_size:
        minibatch_size = batch_size

    # Track metrics across epochs
    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_updates = 0
    
    # Pre-calculate batch.actions mask
    act_mask_full = batch.actions != -1
    safe_actions_full = batch.actions.clone()
    safe_actions_full[~act_mask_full] = 0

    # 2. Optimization Loop with Minibatches
    for epoch in range(update_epochs):
        for mb_inds in generate_minibatches(batch_size, minibatch_size):
            # Slice the tensors for this minibatch
            mb_enc_indices = batch.enc_indices[mb_inds]
            mb_enc_values = batch.enc_values[mb_inds]
            mb_enc_offsets = batch.enc_offsets[mb_inds]
            mb_decoder_inputs = [batch.decoder_inputs[i] for i in mb_inds]
            mb_action_masks = batch.action_masks[mb_inds]
            
            mb_advantages = advantages[mb_inds]
            mb_returns = returns[mb_inds]
            mb_old_log_probs = batch.old_log_probs[mb_inds]
            
            mb_safe_actions = safe_actions_full[mb_inds]
            mb_act_mask = act_mask_full[mb_inds]

            # Forward pass
            logits, new_values = model(mb_enc_indices, mb_enc_values, mb_enc_offsets, mb_decoder_inputs, mb_action_masks)
            
            probs = torch.softmax(logits, dim=-1)
            log_probs_full = torch.log_softmax(logits, dim=-1)
            
            action_log_probs = log_probs_full.gather(1, mb_safe_actions)
            new_log_probs = (action_log_probs * mb_act_mask.float()).sum(dim=1)
            
            masked_probs = probs.masked_fill(mb_action_masks, 0.0)
            masked_log_probs = log_probs_full.masked_fill(mb_action_masks, 0.0)
            entropy = -(masked_probs * masked_log_probs).sum(dim=-1).mean()
            
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            new_values = new_values.squeeze(-1)
            v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
            
            loss = pg_loss - ent_coef * entropy + vf_coef * v_loss
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_pg_loss += pg_loss.detach()
            total_v_loss += v_loss.detach()
            total_entropy += entropy.detach()
            total_approx_kl += approx_kl.detach()
            total_updates += 1
            
    # Calculate Explained Variance (using final predictions over entire batch for accuracy, or just use targets)
    with torch.no_grad():
        _, final_values = model(batch.enc_indices, batch.enc_values, batch.enc_offsets, batch.decoder_inputs, batch.action_masks)
        final_values = final_values.squeeze(-1)
        var_y = torch.var(returns)
        explained_var = (1 - torch.var(returns - final_values) / (var_y + 1e-8)) if var_y > 0 else torch.tensor(0.0)
    
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
    
    return LossMetrics(
        actor_loss=(total_pg_loss / total_updates).item(), 
        critic_loss=(total_v_loss / total_updates).item(), 
        entropy=(total_entropy / total_updates).item(), 
        kl_divergence=(total_approx_kl / total_updates).item(), 
        explained_variance=explained_var.item() if isinstance(explained_var, torch.Tensor) else float(explained_var), 
        grad_norm=grad_norm_val
    )
