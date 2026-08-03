import os
import sys
import torch
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
from typing import List
import json
from datetime import datetime
from pathlib import Path
import time
import collections

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from model_ppo import TitanTransformerPPO, TitanConfig
from ppo_env import PokemonPPOEnv
from ppo_memory import RolloutBuffer
from ppo_core import update_ppo
from ppo_validation import run_validation
from ppo_config import load_config
from ppo_checkpoint import save_checkpoint, load_checkpoint
from ppo_sampler import OpponentSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "01_baseline", "agent")))
from main import read_deck_csv
DECK_LIST = read_deck_csv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    log_dir = Path(os.path.dirname(__file__)) / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "metrics.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config, ppo_cfg, curr_cfg = load_config()
    model = TitanTransformerPPO(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)
    
    sampler = OpponentSampler(curr_cfg)
    start_episode = load_checkpoint(model, optimizer, filename="latest.pt")
    if not start_episode:
        start_episode = 0

    max_seq_len = 2048 # Using fixed safe max sequence length
    memory = RolloutBuffer(
        max_steps=ppo_cfg.max_steps,
        max_seq_len=max_seq_len,
        max_options=config.max_actions,
        device=device
    )    
    total_episodes = start_episode + 2 if args.test_mode else ppo_cfg.total_episodes
    
    episode = start_episode
    global_step = 0
    best_val_win_rate = -1.0
    
    # Episodic Tracking Queues
    ep_length_q = collections.deque(maxlen=100)
    ep_reward_q = collections.deque(maxlen=100)
    win_rate_q = collections.deque(maxlen=100)
    
    while episode < total_episodes:
        if args.test_mode:
            from ppo_bridge import generate_selfplay_script
            opponent_path = generate_selfplay_script("latest.pt")
            opponent_name = "Self-Play (Test Mode)"
        else:
            opponent_path, opponent_name = sampler.sample_opponent()
            
        env = PokemonPPOEnv(opponent=opponent_path)
        ep_start_time = time.perf_counter()
        step_count = 0
        step_result = env.reset()
        done = step_result.done
        
        while not done:
            step_count += 1
            global_step += 1
            parsed = step_result.obs
            
            if parsed is None:
                step_result = env.step(DECK_LIST)
                if step_result.info.get('is_invalid'):
                    print(f"FATAL ENGINE REJECTION (SETUP): Attempted Action: {step_result.info.get('failed_action')}")
                    raise RuntimeError("INVALID step! Action failed engine check during setup.")
                done = step_result.done
                continue

            num_options = parsed.num_options
            max_count = parsed.max_count
            
            # Format inputs
            enc_indices = torch.tensor(parsed.enc_index, dtype=torch.long, device=device).unsqueeze(0)
            enc_values = torch.tensor(parsed.enc_value, dtype=torch.float, device=device).unsqueeze(0)
            enc_offsets = torch.tensor(parsed.enc_offset, dtype=torch.long, device=device).unsqueeze(0)

            dec_inputs_list = []
            for a in range(num_options):
                start = parsed.dec_offset[a] if a < len(parsed.dec_offset) else len(parsed.dec_index)
                end = parsed.dec_offset[a+1] if a+1 < len(parsed.dec_offset) else len(parsed.dec_index)
                idxs = torch.tensor(parsed.dec_index[start:end], dtype=torch.long, device=device)
                vals = torch.tensor(parsed.dec_value[start:end], dtype=torch.float, device=device)
                dec_inputs_list.append((idxs, vals, start))

            action_masks = torch.zeros((1, num_options), dtype=torch.bool, device=device)

            with torch.no_grad():
                logits, value_tensor = model(enc_indices, enc_values, enc_offsets, [dec_inputs_list], action_masks)
                value = value_tensor.squeeze(-1).item()
                
                masked_logits = logits.masked_fill(action_masks, float('-inf'))
                dist = torch.distributions.Categorical(logits=masked_logits)
                
                if max_count > 1 and max_count <= num_options:
                    # Sample topk or multiple without replacement. For PPO we typically sample from dist.
                    # Since we are asked to use topk logic (like validation) for now per user prompt:
                    # "train_sequential.py must do the same during the training rollout."
                    _, topk_actions = torch.topk(masked_logits, k=max_count, dim=-1)
                    action = topk_actions[0].cpu().tolist()
                    
                    # For log prob we need to compute it properly. For this prototype we will 
                    # approximate log_prob as sum of log probs for simplicity, as we use topk.
                    log_prob = dist.log_prob(topk_actions[0]).sum().item()
                else:
                    act = dist.sample()
                    action = [int(act.item())]
                    log_prob = dist.log_prob(act).item()

            step_result = env.step(action)
            done = step_result.done
            
            if step_result.info.get('is_invalid'):
                print(f"FATAL ENGINE REJECTION: Attempted Action: {step_result.info.get('failed_action')}")
                raise RuntimeError(f"INVALID step! Action failed engine check.")
            
            memory.add(parsed, action, step_result.reward, done, value, log_prob)
            
            # PHASE 2: Optimization Trigger
            if memory.ptr == memory.max_steps:
                batch = memory.get_batch()
                next_value = 0.0 # Next value assumed 0 across batch boundaries for simplicity in this TCG unless step_count logic dictates otherwise
                
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    with record_function("model_inference"):
                        loss_metrics = update_ppo(model, optimizer, batch, next_value, clip_coef=ppo_cfg.clip_coef, ent_coef=ppo_cfg.ent_coef, vf_coef=ppo_cfg.vf_coef)
                
                memory.clear()
                
                # PHASE 3: Log Training Metrics
                actor_loss_val = loss_metrics.actor_loss.item() if hasattr(loss_metrics.actor_loss, 'item') else loss_metrics.actor_loss
                critic_loss_val = loss_metrics.critic_loss.item() if hasattr(loss_metrics.critic_loss, 'item') else loss_metrics.critic_loss
                entropy_val = loss_metrics.entropy.item() if hasattr(loss_metrics.entropy, 'item') else loss_metrics.entropy
                kl_val = loss_metrics.kl_divergence.item() if hasattr(loss_metrics.kl_divergence, 'item') else loss_metrics.kl_divergence
                exp_var_val = loss_metrics.explained_variance.item() if hasattr(loss_metrics.explained_variance, 'item') else loss_metrics.explained_variance
                grad_norm_val = loss_metrics.grad_norm.item() if hasattr(loss_metrics.grad_norm, 'item') else loss_metrics.grad_norm
                
                train_data = {
                    "timestamp": datetime.now().isoformat(),
                    "global_step": global_step,
                    "train/pg_loss": actor_loss_val,
                    "train/v_loss": critic_loss_val,
                    "train/entropy": entropy_val,
                    "train/explained_variance": exp_var_val,
                    "train/kl_divergence": kl_val,
                    "train/grad_norm": grad_norm_val
                }
                
                timestamp_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp_str}] [TRAIN] Step {global_step} | PG Loss: {actor_loss_val:.4f} | V Loss: {critic_loss_val:.4f} | Entropy: {entropy_val:.4f}")
                
                with open(log_file, "a") as f:
                    f.write(json.dumps(train_data) + "\n")
            
        # PHASE 3: Episodic Metrics (Tracked in Collection Loop)
        episode += 1
        ep_duration = time.perf_counter() - ep_start_time
        sps = step_count / ep_duration if ep_duration > 0 else 0.0
        
        ep_length_q.append(step_count)
        ep_reward_q.append(step_result.reward)
        win = 1 if step_result.reward > 0 else 0
        win_rate_q.append(win)
        
        ep_data = {
            "timestamp": datetime.now().isoformat(),
            "global_step": global_step,
            "episode": episode,
            "metrics/ep_length": step_count,
            "metrics/ep_reward": step_result.reward,
            "metrics/win_rate_100": sum(win_rate_q) / len(win_rate_q),
            "metrics/sps": round(sps, 2),
            "metrics/duration_sec": round(ep_duration, 4),
            "opponent": opponent_name
        }
        
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp_str}] [EPISODE] Ep {episode} | vs {opponent_name} | Duration: {ep_duration:.2f}s | Steps: {step_count} | Reward: {step_result.reward} | 100-Ep Win Rate: {ep_data['metrics/win_rate_100']*100:.0f}%")
        
        with open(log_file, "a") as f:
            f.write(json.dumps(ep_data) + "\n")

        # Validation trigger
        if episode % curr_cfg.validation_freq == 0 or args.test_mode:
            val_num = 1 if args.test_mode else 5
            val_res = run_validation(model, sampler, num_games=val_num, device=device)
            print(f"Validation Win Rate: {val_res.win_rate:.2f}")
            val_data = {"global_step": global_step, "episode": episode, "val_win_rate": val_res.win_rate}
            with open(log_file, "a") as f:
                f.write(json.dumps(val_data) + "\n")
            
            if val_res.win_rate > best_val_win_rate:
                best_val_win_rate = val_res.win_rate
                save_checkpoint(model, optimizer, episode, "best.pt")
                print(f"New best validation win rate! Saved best.pt")

        # Save checkpoint
        if episode % curr_cfg.checkpoint_freq == 0 or args.test_mode:
            save_checkpoint(model, optimizer, episode, "latest.pt")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stamped_name = f"checkpoint_ep{episode}_{stamp}.pt"
            save_checkpoint(model, optimizer, episode, stamped_name)
            print(f"Saved latest.pt and {stamped_name}")

if __name__ == "__main__":
    main()
