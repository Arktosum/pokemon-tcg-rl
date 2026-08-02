import sys
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime
from typing import List, Tuple

_bc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../02_behavioral_cloning"))
if _bc_dir not in sys.path:
    sys.path.insert(0, _bc_dir)

from dataset import ExpertSample, pad_replay_batch

_ppo_dir = os.path.abspath(os.path.dirname(__file__))
if _ppo_dir not in sys.path:
    sys.path.insert(0, _ppo_dir)

from model_ppo import TitanConfig, TitanTransformerPPO
from env_wrapper import PokemonPPOEnv

_baseline_agent = os.path.abspath(os.path.join(os.path.dirname(__file__), "../01_baseline/agent"))
if _baseline_agent not in sys.path:
    sys.path.insert(0, _baseline_agent)

from parser import get_encoder_input, get_decoder_input
from cg.api import to_observation_class
from main import read_deck_csv

DECK_LIST = read_deck_csv()

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = []
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]
            
        if dones[t]:
            next_val = 0.0
            last_gae_lam = 0.0

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        advantages.insert(0, last_gae_lam)
    
    return advantages

def calc_entropy(masked_logits, action_mask):
    probs = torch.softmax(masked_logits, dim=-1)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    log_probs = torch.where(action_mask, torch.zeros_like(log_probs), log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy

def collect_rollout(env, model, max_steps=10000000, ep=0):
    model.eval()
    samples = []
    rewards = []
    values = []
    dones = []
    log_probs = []

    obs = env.reset()
    for step_i in range(max_steps):
        obs_raw = obs[0]['observation'] if isinstance(obs, list) else obs
        obs_class = to_observation_class(obs_raw)
        
        if obs_class.select is None or obs_class.current is None:
            obs, reward, done, info = env.step(DECK_LIST)
            done_flag = done[0] if isinstance(done, list) else done
            if done_flag:
                rew_val = reward[0] if isinstance(reward, list) else reward
                rew_val = rew_val if rew_val is not None else 0.0
                timestamp = datetime.now().isoformat()
                with open('logs/metrics.jsonl', 'a') as f:
                    f.write(json.dumps({"type": "gameplay", "timestamp": timestamp, "episode": ep, "reward": float(rew_val), "steps": step_i, "win": 1 if rew_val > 0 else 0}) + '\n')
                break
            continue
            
        enc_sv = get_encoder_input(obs_class)
        
        num_options = len(obs_class.select.option) if obs_class.select.option else 1
        if num_options == 0: num_options = 1
        actions = [[i] for i in range(num_options)]
        dec_sv = get_decoder_input(obs_class, actions)
        
        enc_indices = torch.tensor(enc_sv.index, dtype=torch.long).unsqueeze(0)
        enc_values = torch.tensor(enc_sv.value, dtype=torch.float).unsqueeze(0)
        enc_offsets = torch.tensor(enc_sv.offset, dtype=torch.long).unsqueeze(0)
        
        dec_indices = torch.tensor(dec_sv.index, dtype=torch.long)
        dec_values = torch.tensor(dec_sv.value, dtype=torch.float)
        dec_offsets = torch.tensor(dec_sv.offset, dtype=torch.long)
        
        action_mask = torch.zeros((1, num_options), dtype=torch.bool)
        
        with torch.no_grad():
            logits, value = model(enc_indices, enc_values, enc_offsets, dec_indices, dec_values, dec_offsets, action_mask)
            
        dist = torch.distributions.Categorical(logits=logits[0])
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx))
        
        dec_inputs_list = []
        cursor = 0
        for a in range(num_options):
            start = dec_sv.offset[a] if a < len(dec_sv.offset) else len(dec_sv.index)
            end = dec_sv.offset[a+1] if a+1 < len(dec_sv.offset) else len(dec_sv.index)
            
            idxs = torch.tensor(dec_sv.index[start:end], dtype=torch.long)
            vals = torch.tensor(dec_sv.value[start:end], dtype=torch.float)
            dec_inputs_list.append((idxs, vals, start))
            
        sample = ExpertSample(
            encoder_indices=torch.tensor(enc_sv.index, dtype=torch.long),
            encoder_values=torch.tensor(enc_sv.value, dtype=torch.float),
            encoder_offsets=torch.tensor(enc_sv.offset, dtype=torch.long),
            decoder_inputs=dec_inputs_list,
            target=action_idx
        )
        val_item = value.item()

        try:
            obs, reward, done, info = env.step([action_idx])
            done_flag = done[0] if isinstance(done, list) else done
            rew_val = reward[0] if isinstance(reward, list) else reward
        except Exception:
            done_flag = True
            rew_val = 0.0
        
        samples.append(sample)
        rewards.append(rew_val if rew_val is not None else 0.0)
        values.append(val_item)
        dones.append(int(done_flag))
        log_probs.append(log_prob)
        
        if done_flag:
            timestamp = datetime.now().isoformat()
            with open('logs/metrics.jsonl', 'a') as f:
                f.write(json.dumps({"type": "gameplay", "timestamp": timestamp, "episode": ep, "reward": float(rew_val), "steps": step_i, "win": 1 if rew_val > 0 else 0}) + '\n')
            print(f"Episode finished natively after {step_i + 1} steps")
            break
            
    next_value = 0.0
    return samples, rewards, values, dones, log_probs, next_value

def train(test_mode=False):
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    cfg = TitanConfig()
    model = TitanTransformerPPO(cfg)
    
    bc_model_path = os.path.join(_bc_dir, "checkpoints", "titan_bc_best.pt")
    if os.path.exists(bc_model_path):
        state_dict = torch.load(bc_model_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Warm-started from BC model.")
    else:
        print("BC model not found, starting from scratch.")
        
    actor_params = []
    critic_params = []
    for name, p in model.named_parameters():
        if 'critic' in name:
            critic_params.append(p)
        else:
            actor_params.append(p)
            
    optimizer = optim.Adam([
        {'params': actor_params, 'lr': 1e-4},
        {'params': critic_params, 'lr': 3e-5}
    ])
    
    env = PokemonPPOEnv()
    
    best_win_rate = -1.0
    val_env = PokemonPPOEnv(fixed_opponent=os.path.abspath(os.path.join(os.path.dirname(__file__), "../03_rule_based_eval/agents/dragapult.py")))

    episodes = 1 if test_mode else 100
    for ep in range(episodes):
        samples, rewards, values, dones, old_log_probs, next_value = collect_rollout(env, model, max_steps=50 if test_mode else 10000000, ep=ep)
        
        if len(samples) == 0:
            continue
            
        advantages = compute_gae(rewards, values, dones, next_value)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        old_log_probs = torch.stack(old_log_probs).detach()
        
        batch = pad_replay_batch(samples)
        
        model.train()
        logits, new_values = model(
            batch['encoder_indices'], batch['encoder_values'], batch['encoder_offsets'],
            batch['dec_indices'], batch['dec_values'], batch['dec_offsets'],
            batch['action_mask']
        )
        
        entropy = calc_entropy(logits, batch['action_mask'])
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        new_log_probs = dist.log_prob(batch['targets'])
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        actor_loss = -(torch.min(ratio * advantages, clip_adv)).mean()
        
        new_values = new_values.squeeze(-1)
        val_loss_unclipped = (new_values - returns) ** 2
        val_clipped = torch.tensor(values) + torch.clamp(new_values - torch.tensor(values), -0.2, 0.2)
        val_loss_clipped = (val_clipped - returns) ** 2
        critic_loss = 0.5 * torch.max(val_loss_unclipped, val_loss_clipped).mean()
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with open('logs/metrics.jsonl', 'a') as f:
            f.write(json.dumps({"type": "network", "timestamp": datetime.now().isoformat(), "episode": ep, "actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item()), "entropy": float(entropy.item())}) + '\n')
            
        print(f"Ep {ep}: loss {loss.item():.4f}, actor_loss {actor_loss.item():.4f}, critic_loss {critic_loss.item():.4f}, entropy {entropy.item():.4f}")
        
        if (ep + 1) % 50 == 0:
            model.eval()
            val_wins = 0
            val_games = 10
            print("Running validation against Dragapult...")
            for _ in range(val_games):
                obs = val_env.reset()
                for _ in range(1000):
                    obs_raw = obs[0]['observation'] if isinstance(obs, list) else obs
                    obs_class = to_observation_class(obs_raw)
                    if obs_class.select is None or obs_class.current is None:
                        obs, reward, done, info = val_env.step(DECK_LIST)
                        done_flag = done[0] if isinstance(done, list) else done
                        if done_flag:
                            rew_val = reward[0] if isinstance(reward, list) else reward
                            if rew_val is not None and rew_val > 0:
                                val_wins += 1
                            break
                        continue
                    
                    enc_sv = get_encoder_input(obs_class)
                    num_options = len(obs_class.select.option) if obs_class.select.option else 1
                    if num_options == 0: num_options = 1
                    actions = [[i] for i in range(num_options)]
                    dec_sv = get_decoder_input(obs_class, actions)
                    
                    enc_indices = torch.tensor(enc_sv.index, dtype=torch.long).unsqueeze(0)
                    enc_values = torch.tensor(enc_sv.value, dtype=torch.float).unsqueeze(0)
                    enc_offsets = torch.tensor(enc_sv.offset, dtype=torch.long).unsqueeze(0)
                    
                    dec_indices = torch.tensor(dec_sv.index, dtype=torch.long)
                    dec_values = torch.tensor(dec_sv.value, dtype=torch.float)
                    dec_offsets = torch.tensor(dec_sv.offset, dtype=torch.long)
                    
                    action_mask = torch.zeros((1, num_options), dtype=torch.bool)
                    
                    with torch.no_grad():
                        logits, _ = model(enc_indices, enc_values, enc_offsets, dec_indices, dec_values, dec_offsets, action_mask)
                        
                    action_idx = torch.argmax(logits[0]).item()
                    
                    try:
                        obs, reward, done, info = val_env.step([action_idx])
                        done_flag = done[0] if isinstance(done, list) else done
                        rew_val = reward[0] if isinstance(reward, list) else reward
                    except Exception:
                        done_flag = True
                        rew_val = 0.0
                        
                    if done_flag:
                        if rew_val is not None and rew_val > 0:
                            val_wins += 1
                        break
            val_win_rate = val_wins / val_games
            print(f"Validation win rate: {val_win_rate:.2f}")
            if val_win_rate > best_win_rate:
                best_win_rate = val_win_rate
                torch.save(model.state_dict(), 'checkpoints/titan_ppo_best.pt')
                print("New best model saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()
    
    train(test_mode=args.test_mode)
    print("Training finished.")
