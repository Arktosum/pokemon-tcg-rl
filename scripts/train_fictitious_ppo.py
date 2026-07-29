import os
import gc
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from env import PTCGEnv
from model import PokemonActorCritic
from ppo_buffer import PPOBuffer
from greedy_agent import GreedyAgent
from advanced_agents import RandomAgent, AdvancedHeuristicAgent

def get_model_action(model, obs, env, return_log_prob=False):
    state_vec = obs["obs"]
    mask = obs["action_mask"]
    s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
    
    with torch.no_grad():
        policy, _ = model(s_tensor, m_tensor)
        
    p = policy.squeeze(0).numpy()
    valid_actions = np.where(mask == 1)[0]
    
    if len(valid_actions) > 0:
        p_valid = p[valid_actions]
        if p_valid.sum() > 0:
            p_valid /= p_valid.sum()
            action = int(np.random.choice(valid_actions, p=p_valid))
        else:
            action = int(np.random.choice(valid_actions))
    else:
        action = 0
        
    if return_log_prob:
        return action, 0.0 # Not used for frozen inference
    return action

def get_frozen_kl(active_policy, frozen_policy, masks):
    # active_policy: (B, num_actions)
    # frozen_policy: (B, num_actions)
    active_probs = torch.clamp(active_policy, 1e-8, 1.0) * masks
    active_probs = active_probs / (active_probs.sum(dim=1, keepdim=True) + 1e-8)
    
    frozen_probs = torch.clamp(frozen_policy, 1e-8, 1.0) * masks
    frozen_probs = frozen_probs / (frozen_probs.sum(dim=1, keepdim=True) + 1e-8)
    
    # KL(P_active || P_frozen) = sum P_active * log(P_active / P_frozen)
    kl = torch.sum(active_probs * (torch.log(active_probs + 1e-8) - torch.log(frozen_probs + 1e-8)) * masks, dim=1)
    return kl.mean()

def run_fictitious_ppo(args):
    env = PTCGEnv()
    
    model = PokemonActorCritic()
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TOP_ELO_BC_MODEL_FINAL.pt")
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print(f"Loaded BC final weights from {ckpt_path}.")
    else:
        print(f"Error: BC Final checkpoint not found at {ckpt_path}")
        return
        
    # Unfreeze all parameters, including value head
    for param in model.parameters():
        param.requires_grad = True
        
    # SMOKE TEST GATE 1 - Gradient check
    if args.smoke_test:
        print("=== GATE 1: GRADIENT FLOW CHECK ===")
        vh_param = model.value_head[0].weight
        print(f"Value Head requires_grad BEFORE dummy pass: {vh_param.requires_grad}")
        print(f"Value Head grad BEFORE dummy pass: {vh_param.grad}")
        
        dummy_state = torch.zeros((1, 120), dtype=torch.float32)
        dummy_mask = torch.ones((1, 500), dtype=torch.int8)
        
        p, v = model(dummy_state, dummy_mask)
        loss = v.sum()
        loss.backward()
        
        print(f"Value Head grad AFTER dummy pass: {vh_param.grad is not None and torch.sum(torch.abs(vh_param.grad)) > 0}")
        model.zero_grad()
        print("===================================\n")
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    buffer = PPOBuffer()
    
    # Frozen Reference Model
    frozen_ref = PokemonActorCritic()
    frozen_ref.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    for param in frozen_ref.parameters():
        param.requires_grad = False
    frozen_ref.eval()
    
    if args.smoke_test:
        print("=== GATE 1: FROZEN REFERENCE CHECK ===")
        print(f"Frozen Ref requires_grad: {any(p.requires_grad for p in frozen_ref.parameters())}")
        print("======================================\n")

    # Opponents
    greedy_agent = GreedyAgent()
    adv_agent = AdvancedHeuristicAgent()
    random_agent = RandomAgent()
    
    num_episodes = args.episodes
    clip_param = 0.2
    entropy_coef = 0.05
    value_coef = 0.5
    kl_coef = 0.02
    ppo_epochs = 4
    
    opponent_counts = Counter()
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        
        # Opponent Selection
        rand = random.random()
        if rand < 0.40:
            opp_name = "Greedy"
            opp_agent = greedy_agent
        elif rand < 0.70:
            opp_name = "FrozenBC"
            opp_agent = frozen_ref
        elif rand < 0.90:
            opp_name = "Advanced"
            opp_agent = adv_agent
        else:
            opp_name = "Random"
            opp_agent = random_agent
            
        opponent_counts[opp_name] += 1
        
        # P1 is Model, P2 is Opponent
        p1_idx = 0
        
        while not done and step < 200:
            state_vec = obs["obs"]
            mask = obs["action_mask"]
            current_player = int(state_vec[2])
            
            if current_player != p1_idx:
                # Opponent acts
                if opp_name == "FrozenBC":
                    action = get_model_action(opp_agent, obs, env)
                else:
                    if hasattr(opp_agent, 'act'):
                        try:
                            action = opp_agent.act(obs, env)
                        except TypeError:
                            action = opp_agent.act(obs)
                    else:
                        action = 0
                
            else:
                # P1 (Our learning model) acts
                s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
                
                with torch.no_grad():
                    policy, value = model(s_tensor, m_tensor)
                    
                p = policy.squeeze(0)
                valid_actions = np.where(mask == 1)[0]
                if len(valid_actions) > 0:
                    p_valid = p[valid_actions]
                    p_sum = p_valid.sum()
                    if p_sum > 0:
                        p_valid /= p_sum
                        dist = torch.distributions.Categorical(p_valid)
                        action_idx = dist.sample()
                        action = int(valid_actions[action_idx.item()])
                        log_prob = dist.log_prob(action_idx).item()
                    else:
                        action = int(np.random.choice(valid_actions))
                        log_prob = -np.log(len(valid_actions))
                else:
                    break
                
                value_val = value.item()
                
            try:
                next_obs, reward, is_done, _, _ = env.step(action)
            except Exception as e:
                print(f"Exception during step in episode {episode}: {e}")
                break
                
            done = is_done or env.is_done
            
            if current_player == p1_idx:
                buffer.store(state_vec, mask, action, reward, value_val, log_prob, 1 - int(done))
            
            obs = next_obs
            step += 1
            
        # PPO Update if we have experiences
        if len(buffer.rewards) > 0:
            advantages, returns = buffer.compute_gae(0.0)
            
            b_states = torch.tensor(np.array(buffer.states), dtype=torch.float32)
            b_masks = torch.tensor(np.array(buffer.action_masks), dtype=torch.float32)
            b_actions = torch.tensor(buffer.actions, dtype=torch.long)
            b_old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
            b_returns = torch.tensor(returns, dtype=torch.float32)
            b_advantages = torch.tensor(advantages, dtype=torch.float32)
            
            if len(b_advantages) > 1:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            epoch_act_loss = []
            epoch_crit_loss = []
            epoch_kl = []
            
            for _ in range(ppo_epochs):
                policy, values = model(b_states, b_masks)
                values = values.squeeze(1)
                
                clamped_policy = torch.clamp(policy, 1e-8, 1.0)
                
                # KL Divergence vs Frozen Reference
                with torch.no_grad():
                    frozen_p, _ = frozen_ref(b_states, b_masks)
                
                kl_div = get_frozen_kl(policy, frozen_p, b_masks)
                
                # Entropy
                valid_probs = clamped_policy * b_masks
                valid_probs = valid_probs / (valid_probs.sum(dim=1, keepdim=True) + 1e-8)
                log_probs_valid = torch.log(valid_probs + 1e-8)
                entropy = -torch.sum(valid_probs * log_probs_valid * b_masks, dim=1).mean()
                
                dist = torch.distributions.Categorical(clamped_policy)
                new_log_probs = dist.log_prob(b_actions)
                
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, b_returns)
                
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy + kl_coef * kl_div
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_act_loss.append(actor_loss.item())
                epoch_crit_loss.append(critic_loss.item())
                epoch_kl.append(kl_div.item())
                
            last_actor_loss = np.mean(epoch_act_loss)
            last_critic_loss = np.mean(epoch_crit_loss)
            last_kl = np.mean(epoch_kl)
            
            buffer.clear()
            
            if args.smoke_test:
                print(f"Ep {episode:03d} vs {opp_name:10s} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f} | KL: {last_kl:.6f} | Steps: {step}")
            else:
                if episode % 10 == 0:
                    print(f"Ep {episode:05d} vs {opp_name:10s} | Actor Loss: {last_actor_loss:.4f} | Critic Loss: {last_critic_loss:.4f} | KL: {last_kl:.6f}", flush=True)

        if not args.smoke_test and episode % (num_episodes // 10) == 0:
            # Checkpoint every 10% (actually instructions say every 5 epochs... wait! Epochs in PPO usually means full passes, but the prompt says 'Checkpoint every 5 epochs to checkpoints/TITAN_FICTITIOUS_PPO_01_epoch_XXX.pt'. Here 'epoch' likely means 'every 5 full passes of the dataset', but in RL it's episodes. Let's checkpoint every 5 episodes? Or maybe 3000 episodes is meant. The user says 'every 5 epochs'. In RL, epoch could be a chunk of episodes or the PPO updates. I will checkpoint every 5 episodes just to be safe, but wait, 3000/5 = 600 checkpoints. That's too many. Let's assume an 'epoch' is 100 episodes, so every 500 episodes. Wait, user says 'Checkpoint every 5 epochs'. Let's just create 'epoch' variable that ticks every 5 episodes? Or maybe 'epoch' is just what I call the outer loop? Let's assume the outer loop is episodes, so I will checkpoint every 500 episodes.)
            # Wait, user explicitly says: "Checkpoint every 5 epochs to checkpoints/TITAN_FICTITIOUS_PPO_01_epoch_XXX.pt"
            # In my previous scripts, did I use epochs? No, num_episodes. I'll define an 'epoch' as 100 episodes.
            pass
            
        if not args.smoke_test and episode % 500 == 0:
            epoch_num = episode // 100
            chk_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", f"TITAN_FICTITIOUS_PPO_01_epoch_{epoch_num:03d}.pt")
            torch.save({'model_state_dict': model.state_dict()}, chk_path)

    if args.smoke_test:
        print("\n=== SMOKE TEST: OPPONENT COUNTS ===")
        for k, v in opponent_counts.items():
            print(f"{k}: {v}")
        print("===================================\n")
    else:
        # Save final weights
        final_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "TITAN_FICTITIOUS_PPO_FINAL.pt")
        torch.save({'model_state_dict': model.state_dict()}, final_path)
        print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    
    # User said: Checkpoint every 5 epochs. Let me implement a proper epoch logic if needed.
    run_fictitious_ppo(args)
