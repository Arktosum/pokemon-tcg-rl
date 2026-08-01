import os
import sys
import glob
import time
import csv
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from datetime import datetime
import numpy as np
import multiprocessing as mp
import collections

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))
from vector_env import VectorEnv
from rollout_buffer import RolloutBuffer
from model import TitanTransformer
from random_agent import random_agent

# Configuration
NUM_ENVS = 16
STEPS_PER_ENV = 128
PPO_STEPS = NUM_ENVS * STEPS_PER_ENV
OPPONENT_POOL = ["random_agent", "setup_agent", "greedy_agent", "tactical_agent", "self_play_agent", "aggro_agent", "heuristic_agent"]
PPO_EPOCHS = 4
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LEARNING_RATE = 3e-5
MAX_UPDATES = 1000

# We intentionally do not load any checkpoints so the agent trains from scratch
def get_latest_checkpoint():
    return None, None

def collate_observations(obs_list):
    enc_indices, enc_weights, enc_offsets = [], [], []
    dec_indices, dec_weights, dec_offsets = [], [], []
    legal_counts = []
    
    enc_idx_offset = 0
    dec_idx_offset = 0
    
    for obs in obs_list:
        ei = torch.tensor(obs["enc_indices"], dtype=torch.int32)
        ew = torch.tensor(obs["enc_weights"], dtype=torch.float32)
        eo = torch.tensor(obs["enc_offsets"], dtype=torch.int32)
        
        enc_indices.append(ei)
        enc_weights.append(ew)
        enc_offsets.append(eo + enc_idx_offset)
        enc_idx_offset += len(ei)
        
        di = torch.tensor(obs["dec_indices"], dtype=torch.int32)
        dw = torch.tensor(obs["dec_weights"], dtype=torch.float32)
        do = torch.tensor(obs["dec_offsets"], dtype=torch.int32)
        
        dec_indices.append(di)
        dec_weights.append(dw)
        dec_offsets.append(do + dec_idx_offset)
            
        dec_idx_offset += len(di)
        legal_counts.append(obs["legal_count"])
        
    return {
        "enc_indices": torch.cat(enc_indices) if enc_indices else torch.empty(0, dtype=torch.int32),
        "enc_offsets": torch.cat(enc_offsets) if enc_offsets else torch.empty(0, dtype=torch.int32),
        "enc_weights": torch.cat(enc_weights) if enc_weights else torch.empty(0, dtype=torch.float32),
        "dec_indices": torch.cat(dec_indices) if dec_indices else torch.empty(0, dtype=torch.int32),
        "dec_offsets": torch.cat(dec_offsets) if dec_offsets else torch.empty(0, dtype=torch.int32),
        "dec_weights": torch.cat(dec_weights) if dec_weights else torch.empty(0, dtype=torch.float32),
        "legal_counts": torch.tensor(legal_counts, dtype=torch.int32)
    }

def select_actions(logits):
    probs = F.softmax(logits, dim=-1)
    dist = Categorical(probs)
    actions = dist.sample()
    return actions.tolist(), dist.log_prob(actions).tolist()

def log_to_csv(filepath, metrics_dict):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PPO] Initializing TitanTransformer on {device}...")
    
    model = TitanTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    global_update_count = 0
    global_episodes = 0
    global_steps = 0
    opponent_name = "RandomBot"
    
    ckpt_path, ckpt_type = get_latest_checkpoint()
    if ckpt_path:
        print(f"[PPO] Loading {ckpt_type} weights from: {os.path.basename(ckpt_path)}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if ckpt_type == "PPO":
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                global_update_count = checkpoint.get("update_count", 0)
                global_episodes = checkpoint.get("episodes", 0)
                global_steps = checkpoint.get("total_steps", 0)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("[PPO] WARNING: No checkpoint found. Starting from scratch.")
        
    print(f"[PPO] Initializing {NUM_ENVS} Vector Environments with {opponent_name}...")
    env = VectorEnv(random_agent, num_envs=NUM_ENVS)
    buffer = RolloutBuffer() # Note: RolloutBuffer will store unrolled transitions for simplicity
    
    obs_list = env.reset()
    
    ep_rewards = [0.0] * NUM_ENVS
    ep_lengths = [0] * NUM_ENVS
    
    batch_rewards = []
    batch_lengths = []
    
    global_wins_queue = collections.deque(maxlen=100)
    global_matches_queue = collections.deque(maxlen=100)
    
    start_time = time.time()
    csv_log_path = os.path.join(os.path.dirname(__file__), "metrics.csv")
    
    try:
        while global_update_count < MAX_UPDATES:
            # We no longer reset the env per epoch! We let it run continuously.
            opponent_name = np.random.choice(OPPONENT_POOL)
            print(f"[PPO] Epoch {global_update_count + 1} starting...")
            # Removed obs_list = env.reset() here to avoid Eternal Early Game
            
            model.eval()
            rollout_start_time = time.time()
            
            for _ in range(STEPS_PER_ENV):
                col_obs = collate_observations(obs_list)
                enc_i = col_obs["enc_indices"].to(device)
                enc_o = col_obs["enc_offsets"].to(device)
                enc_w = col_obs["enc_weights"].to(device)
                dec_i = col_obs["dec_indices"].to(device)
                dec_o = col_obs["dec_offsets"].to(device)
                dec_w = col_obs["dec_weights"].to(device)
                legal_counts = col_obs["legal_counts"].to(device)
                
                with torch.no_grad():
                    logits, values = model(enc_i, enc_o, enc_w, dec_i, dec_o, dec_w, legal_counts)
                    actions, log_probs = select_actions(logits)
                
                next_obs_list, rewards, dones, infos = env.step(actions)
                
                for i in range(NUM_ENVS):
                    ep_rewards[i] += rewards[i]
                    ep_lengths[i] += 1
                    global_steps += 1
                    
                    # Store transitions unrolled
                    buffer.add(obs_list[i], actions[i], rewards[i], dones[i], log_probs[i], values[i].item())
                    
                    if dones[i]:
                        global_episodes += 1
                        batch_rewards.append(ep_rewards[i])
                        batch_lengths.append(ep_lengths[i])
                        
                        engine_reward = infos[i].get("engine_reward", 0)
                        global_matches_queue.append(1)
                        if engine_reward == 1:
                            global_wins_queue.append(1)
                        else:
                            global_wins_queue.append(0)
                            
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0
                        # VectorEnv auto-resets, the next_obs_list[i] is already the reset state
                        
                obs_list = next_obs_list
                
            rollout_duration = time.time() - rollout_start_time
            
            # --- PPO UPDATE ---
            optim_start_time = time.time()
            
            # Compute next_values for GAE
            col_obs = collate_observations(obs_list)
            enc_i = col_obs["enc_indices"].to(device)
            enc_o = col_obs["enc_offsets"].to(device)
            enc_w = col_obs["enc_weights"].to(device)
            dec_i = col_obs["dec_indices"].to(device)
            dec_o = col_obs["dec_offsets"].to(device)
            dec_w = col_obs["dec_weights"].to(device)
            legal_counts = col_obs["legal_counts"].to(device)
            
            with torch.no_grad():
                _, next_values = model(enc_i, enc_o, enc_w, dec_i, dec_o, dec_w, legal_counts)
            
            # Reconstruct batch to unroll GAE correctly per environment
            # Our buffer has elements stored linearly: env0_step0, env1_step0, ... envN_step0, env0_step1...
            # To compute GAE properly, we need to extract trajectories per environment
            # This is complex. We will approximate GAE by treating the buffer linearly as N trajectories if we stride them.
            # Actually, RolloutBuffer add() appended linearly.
            # A simpler way is just to recalculate GAE ignoring environment boundaries, but capping at dones.
            # Which is exactly what compute_returns_and_advantages does!
            
            # Wait! The buffer currently has elements interweaved!
            # We must un-interweave them to calculate GAE correctly.
            all_returns = []
            all_advantages = []
            
            for env_idx in range(NUM_ENVS):
                env_rewards = [buffer.rewards[i] for i in range(env_idx, len(buffer.rewards), NUM_ENVS)]
                env_dones = [buffer.dones[i] for i in range(env_idx, len(buffer.dones), NUM_ENVS)]
                env_values = [buffer.values[i] for i in range(env_idx, len(buffer.values), NUM_ENVS)]
                
                next_val = next_values[env_idx].item()
                
                ret = []
                adv = []
                gae = 0
                for step in reversed(range(len(env_rewards))):
                    delta = env_rewards[step] + GAMMA * next_val * (1 - env_dones[step]) - env_values[step]
                    gae = delta + GAMMA * GAE_LAMBDA * (1 - env_dones[step]) * gae
                    adv.insert(0, gae)
                    ret.insert(0, gae + env_values[step])
                    next_val = env_values[step]
                    
                all_returns.append(ret)
                all_advantages.append(adv)
            
            # Flatten lists. They will be ordered env0_step0..env0_stepN, env1_step0..env1_stepN
            # We must re-interweave them to match the buffer order.
            returns_flat = []
            advantages_flat = []
            for step in range(STEPS_PER_ENV):
                for env_idx in range(NUM_ENVS):
                    returns_flat.append(all_returns[env_idx][step])
                    advantages_flat.append(all_advantages[env_idx][step])
                    
            returns = torch.tensor(returns_flat, dtype=torch.float32, device=device)
            advantages = torch.tensor(advantages_flat, dtype=torch.float32, device=device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
            actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
            
            model.train()
            
            avg_p_loss, avg_v_loss, avg_entropy, approx_kl = 0, 0, 0, 0
            
            batch_size = BATCH_SIZE
            
            for epoch in range(PPO_EPOCHS):
                # Re-shuffle every epoch — standard PPO correctness
                indices = torch.randperm(len(buffer.rewards)).tolist()
                chunks = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
                for chunk in chunks:
                    mb_obs = []
                    mb_old_log_probs = []
                    mb_actions = []
                    mb_adv = []
                    mb_ret = []
                    
                    for idx in chunk:
                        legal_count = len(buffer.dec_offsets[idx])
                        mb_obs.append({
                            "enc_indices": buffer.enc_indices[idx],
                            "enc_offsets": buffer.enc_offsets[idx],
                            "enc_weights": buffer.enc_weights[idx],
                            "dec_indices": buffer.dec_indices[idx],
                            "dec_offsets": buffer.dec_offsets[idx],
                            "dec_weights": buffer.dec_weights[idx],
                            "legal_count": legal_count
                        })
                        
                    chunk_t = torch.tensor(chunk, dtype=torch.long, device=device)
                    mb_old_log_probs = old_log_probs[chunk_t]
                    mb_actions = actions[chunk_t]
                    mb_adv = advantages[chunk_t]
                    mb_ret = returns[chunk_t]
                        
                    col_obs = collate_observations(mb_obs)
                    enc_i = col_obs["enc_indices"].to(device)
                    enc_o = col_obs["enc_offsets"].to(device)
                    enc_w = col_obs["enc_weights"].to(device)
                    dec_i = col_obs["dec_indices"].to(device)
                    dec_o = col_obs["dec_offsets"].to(device)
                    dec_w = col_obs["dec_weights"].to(device)
                    legal_counts = col_obs["legal_counts"].to(device)
                    
                    logits, values = model(enc_i, enc_o, enc_w, dec_i, dec_o, dec_w, legal_counts)
                    
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    
                    a = mb_actions
                    new_log_probs = dist.log_prob(a)
                    entropy = dist.entropy().mean()
                    
                    old_lp = mb_old_log_probs
                    ratio = torch.exp(new_log_probs - old_lp)
                    
                    adv = mb_adv
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * adv
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    ret_t = mb_ret
                    critic_loss = F.mse_loss(values.squeeze(-1), ret_t)
                    
                    loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    
                    avg_p_loss += actor_loss.item()
                    avg_v_loss += critic_loss.item()
                    avg_entropy += entropy.item()
                    with torch.no_grad():
                        approx_kl += (old_lp - new_log_probs).mean().item()
            
            total_optim_steps = PPO_EPOCHS * len(chunks)
            avg_p_loss /= total_optim_steps
            avg_v_loss /= total_optim_steps
            avg_entropy /= total_optim_steps
            approx_kl /= total_optim_steps
            
            buffer.clear()
            optim_duration = time.time() - optim_start_time
            
            global_update_count += 1
            
            avg_rew = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            avg_len = sum(batch_lengths) / len(batch_lengths) if batch_lengths else 0
            
            # Use rolling queues for win rate
            total_wins = sum(global_wins_queue)
            total_matches = sum(global_matches_queue)
            win_rate = (total_wins / total_matches * 100) if total_matches else 0
            
            sps = PPO_STEPS / (rollout_duration + optim_duration)
            
            elapsed = time.time() - start_time
            time_per_update = elapsed / global_update_count
            eta_seconds = time_per_update * (MAX_UPDATES - global_update_count)
            
            eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m remaining"
            elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            dashboard = f"""
=========================================================
| Metric                       | Value                  |
=========================================================
| Time / Timestamp             | {timestamp_str:<22} |
| Time / Total Elapsed         | {elapsed_str:<22} |
| Time / Rollout Phase         | {rollout_duration:<20.1f}s |
| Time / Optimization Phase    | {optim_duration:<20.1f}s |
| Time / Steps Per Second      | {sps:<22.1f} |
|------------------------------|------------------------|
| Global / Progress            | Update {global_update_count} / {MAX_UPDATES} ({global_update_count/MAX_UPDATES*100:.1f}%) |
| Global / ETA                 | {eta_str:<22} |
| Global / Opponent Bot        | {opponent_name:<22} |
| Global / Total Env Steps     | {global_steps:<22} |
| Global / Total Episodes      | {global_episodes:<22} |
|------------------------------|------------------------|
| Rollout / Avg Reward         | {avg_rew:<22.4f} |
| Rollout / Avg Ep Length      | {avg_len:<22.1f} |
| Rollout / Win Rate (Batch)   | {win_rate:<21.1f}% |
|------------------------------|------------------------|
| Loss / Policy (Actor)        | {avg_p_loss:<22.4f} |
| Loss / Value (Critic)        | {avg_v_loss:<22.4f} |
| Loss / Entropy               | {avg_entropy:<22.4f} |
| Loss / Approx KL Divergence  | {approx_kl:<22.4f} |
=========================================================
"""
            print(dashboard)
            
            metrics = {
                "timestamp": timestamp_str,
                "update_count": global_update_count,
                "episodes": global_episodes,
                "total_steps": global_steps,
                "opponent": opponent_name,
                "avg_reward": avg_rew,
                "avg_length": avg_len,
                "win_rate": win_rate,
                "actor_loss": avg_p_loss,
                "critic_loss": avg_v_loss,
                "entropy": avg_entropy,
                "kl_div": approx_kl,
                "sps": sps
            }
            log_to_csv(csv_log_path, metrics)
            
            batch_rewards = []
            batch_lengths = []
            
            if global_update_count % 10 == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(os.path.dirname(__file__), f"{ts}_ppo_checkpoint.pt")
                torch.save({
                    "update_count": global_update_count,
                    "episodes": global_episodes,
                    "total_steps": global_steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, save_path)
                print(f"[{timestamp_str}] [CHECKPOINT SAVED] -> {save_path}")
                
    finally:
        env.close()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
