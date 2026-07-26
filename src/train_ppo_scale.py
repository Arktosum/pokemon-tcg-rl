import sys
import os
import time
import random
import glob
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)
sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from cg.api import to_observation_class
from src.model.transformer_policy import MyModel, get_encoder_input, get_decoder_input
from src.env.tcg_env import PokemonTCGEnv
from src.model.heuristic_bot import agent as rule_agent

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==========================================
# CHECKPOINT MANAGER
# ==========================================
class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, episode, win_rate, filename=None):
        if filename is None:
            filename = f"model_ep{episode}_wr{win_rate:.2f}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'win_rate': win_rate
        }, path)
        logger.info(f"[SAVE] Checkpoint safely saved: {path}")

    def load_random_opponent(self, device, d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "*.pt"))
        if not checkpoints:
            return None
            
        chosen_ckpt = random.choice(checkpoints)
        model = MyModel(d_model=d_model, num_heads=num_heads, d_feedforward=d_feedforward, 
                        num_layers_encoder=num_layers_encoder, num_layers_decoder=num_layers_decoder).to(device)
        
        checkpoint = torch.load(chosen_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.debug(f"Loaded opponent from {chosen_ckpt}")
        return model

# ==========================================
# PPO MATHEMATICS
# ==========================================
def compute_gae(rewards, values, gamma=0.99, lam=0.95, bootstrap_value=0.0):
    """Generalized Advantage Estimation.
    
    Args:
        rewards: list of per-step rewards.
        values: list of critic value estimates for each step.
        gamma: discount factor.
        lam: GAE lambda.
        bootstrap_value: If the episode was TRUNCATED (not terminated),
                         this should be V(s_final) from the critic.
                         If the episode TERMINATED naturally, this is 0.0.
    """
    advantages = []
    gae = 0
    values = values + [bootstrap_value]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

# ==========================================
# CORE TRAINING LOOP
# ==========================================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[INIT] Initializing Scaled PPO Trainer on device: {device}")
    
    deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    env = PokemonTCGEnv(deck)
    ckpt_manager = CheckpointManager()
    
    # Model
    model_params = {'d_model': 128, 'num_heads': 2, 'd_feedforward': 256, 'num_layers_encoder': 1, 'num_layers_decoder': 1}
    model = MyModel(**model_params).to(device)
    
    if args.load_bc:
        bc_path = os.path.join("checkpoints", "model_bc.pt")
        if os.path.exists(bc_path):
            logger.info(f"Loading Behavioral Cloning weights from {bc_path}")
            checkpoint = torch.load(bc_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning(f"Could not find BC weights at {bc_path}. Initializing randomly.")
            
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Resume from checkpoint
    start_episode = 1
    if args.resume:
        checkpoints = glob.glob(os.path.join('checkpoints', 'model_ep*.pt'))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            logger.info(f"[RESUME] Loading checkpoint: {latest}")
            ckpt = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_episode = ckpt.get('episode', 0) + 1
            logger.info(f"[RESUME] Resuming from episode {start_episode}")
        else:
            logger.warning("[RESUME] No checkpoints found. Starting from scratch.")
    
    clip_ratio = 0.2
    epochs_per_update = 2
    
    total_episodes = args.episodes
    save_freq = args.save_freq
    
    wins = 0
    draws = 0
    losses = 0
    crashes = 0
    total_steps_taken = 0
    start_time_global = time.time()
    
    # Profiler Context
    prof = None
    if args.profile:
        logger.info("[PROFILE] PyTorch Profiler ENABLED. Will profile the episodes.")
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

    logger.info(f"Starting training for {total_episodes} episodes. Saving every {save_freq} episodes.")

    for episode in range(start_episode, total_episodes + 1):
        ep_start_time = time.time()
        obs_dict = env.reset()
        
        states_enc, states_dec, actions, log_probs_old, values, rewards = [], [], [], [], [], []
        truncated = False
        termination = 'ongoing'
        
        # 50/50 Opponent Sampling
        opponent_model = None
        if random.random() < 0.5:
            opponent_type = "Heuristic Bot"
        else:
            opponent_model = ckpt_manager.load_random_opponent(device, **model_params)
            if opponent_model is None:
                opponent_type = "Heuristic Bot"
            else:
                opponent_type = "Past Checkpoint"
                
        step_count = 0
        
        while True:
            if not obs_dict:
                break
                
            your_index = obs_dict['current']['yourIndex']
            
            # ---------------------------
            # PLAYER 0: PPO AGENT
            # ---------------------------
            if your_index == 0:
                obs = to_observation_class(obs_dict)
                sv_enc = get_encoder_input(obs, env.deck)
                legal_actions = [[i] for i in range(len(obs.select.option))]
                sv_dec = get_decoder_input(obs, legal_actions)
                
                enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32, device=device)
                enc_val = torch.tensor(sv_enc.value, dtype=torch.float32, device=device)
                enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32, device=device)
                dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32, device=device)
                dec_val = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
                dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
                
                with torch.no_grad():
                    val, logits = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
                    dist = torch.distributions.Categorical(logits=logits)
                    action_idx = dist.sample()
                    log_prob = dist.log_prob(action_idx)
                
                chosen_action = legal_actions[action_idx.item()]
                
                states_enc.append((enc_idx, enc_val, enc_off))
                states_dec.append((dec_idx, dec_val, dec_off))
                actions.append(action_idx.item())
                log_probs_old.append(log_prob.item())
                values.append(val.item())
                
                obs_dict, reward, done, info = env.step(chosen_action)
                
                if done:
                    r_arr = [0.0] * len(actions)
                    r_arr[-1] = reward
                    rewards = r_arr
                    termination = info['termination']
                    break
                    
            # ---------------------------
            # PLAYER 1: OPPONENT
            # ---------------------------
            else:
                if opponent_model is not None:
                    # Past Checkpoint
                    obs = to_observation_class(obs_dict)
                    sv_enc = get_encoder_input(obs, env.deck)
                    legal_actions = [[i] for i in range(len(obs.select.option))]
                    sv_dec = get_decoder_input(obs, legal_actions)
                    
                    enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32, device=device)
                    enc_val = torch.tensor(sv_enc.value, dtype=torch.float32, device=device)
                    enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32, device=device)
                    dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32, device=device)
                    dec_val = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
                    dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
                    
                    with torch.no_grad():
                        _, logits = opponent_model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
                        dist = torch.distributions.Categorical(logits=logits)
                        action_idx = dist.sample()
                    chosen_action = legal_actions[action_idx.item()]
                else:
                    # Heuristic Bot
                    try:
                        chosen_action = rule_agent(obs_dict)
                    except Exception:
                        r_arr = [0.0] * len(actions)
                        if len(r_arr) > 0: r_arr[-1] = 1.0 # Bot crashed, we win
                        rewards = r_arr
                        break
                        
                obs_dict, reward, done, info = env.step(chosen_action)
                if done:
                    r_arr = [0.0] * len(actions)
                    if len(r_arr) > 0: r_arr[-1] = reward
                    rewards = r_arr
                    termination = info['termination']
                    break
                    
            step_count += 1
            if step_count > 500:
                # TRUNCATION: Game did not end naturally.
                # Bootstrap the value of the current state from the Critic,
                # instead of treating it as terminal (0.0).
                truncated = True
                termination = 'truncated'
                rewards = [0.0] * len(actions)
                break
                
        # ==========================================
        # OPTIMIZATION
        # ==========================================
        if len(rewards) == 0:
            logger.warning(f"Episode {episode} crashed instantly. Skipping.")
            continue
            
        final_reward = rewards[-1] if not truncated else 0.0
        if termination == 'crash':
            crashes += 1
        elif final_reward > 0:
            wins += 1
        elif final_reward < 0:
            losses += 1
        else:
            draws += 1
            
        # Compute bootstrap value for truncated episodes
        bootstrap_val = 0.0
        if truncated and len(states_enc) > 0:
            # Ask the Critic: "What do you think this state is worth?"
            enc_idx, enc_val, enc_off = states_enc[-1]
            dec_idx, dec_val, dec_off = states_dec[-1]
            with torch.no_grad():
                v_bootstrap, _ = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
                bootstrap_val = v_bootstrap.item()
                
        advantages, returns = compute_gae(rewards, values, bootstrap_value=bootstrap_val)
        total_ploss = 0.0
        total_vloss = 0.0
        total_entropy = 0.0
        
        for ppo_epoch in range(epochs_per_update):
            for i in range(len(actions)):
                enc_idx, enc_val, enc_off = states_enc[i]
                dec_idx, dec_val, dec_off = states_dec[i]
                old_log_prob = log_probs_old[i]
                adv = advantages[i]
                ret = returns[i]
                act = actions[i]
                
                val, logits = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(torch.tensor(act, device=device))
                entropy = dist.entropy()
                
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2)
                value_loss = 0.5 * (val.squeeze() - ret)**2
                loss = policy_loss + value_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                total_entropy += entropy.item()
                
        total_steps_taken += step_count
        ep_duration = time.time() - ep_start_time
        
        # Logging
        expected_score = (wins + 0.5 * draws) / episode
        avg_ploss = total_ploss / (epochs_per_update * len(actions))
        avg_vloss = total_vloss / (epochs_per_update * len(actions))
        avg_entropy = total_entropy / (epochs_per_update * len(actions))
        
        # ANSI Colors
        RESET = "\033[0m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        MAGENTA = "\033[35m"

        # Thresholds
        c_opp = CYAN if opponent_type == "Heuristic Bot" else MAGENTA
        
        if final_reward > 0: c_rew = GREEN
        elif final_reward < 0: c_rew = RED
        else: c_rew = YELLOW
        
        if avg_ploss > 0.5: c_ploss = RED
        elif avg_ploss < -0.5: c_ploss = GREEN
        else: c_ploss = RESET
        
        if expected_score > 0.5: c_win = GREEN
        elif expected_score < 0.3: c_win = RED
        else: c_win = YELLOW
        
        logger.info(f"Ep {episode:04d}/{total_episodes} | "
                    f"Opp: {c_opp}{opponent_type:<15}{RESET} | "
                    f"Reward: {c_rew}{final_reward:>4.1f}{RESET} | "
                    f"Steps: {step_count:03d} | "
                    f"PLoss: {c_ploss}{avg_ploss:>7.4f}{RESET} | "
                    f"VLoss: {avg_vloss:>7.4f} | "
                    f"Ent: {avg_entropy:>6.4f} | Time: {ep_duration:>4.1f}s | "
                    f"W/D/L/C: {GREEN}{wins}{RESET}/{YELLOW}{draws}{RESET}/{RED}{losses}{RESET}/{MAGENTA}{crashes}{RESET} | "
                    f"Score: {c_win}{expected_score:.1%}{RESET}")
                    
        # Checkpointing
        if episode % save_freq == 0:
            ckpt_manager.save_checkpoint(model, optimizer, episode, expected_score)
            
        if prof is not None:
            prof.step()

    total_time = time.time() - start_time_global
    logger.info(f"[DONE] Training Complete! {total_episodes} episodes in {total_time/60:.1f} minutes.")
    logger.info(f"Final Win Rate: {wins/total_episodes:.1%}")
    
    # Final save
    ckpt_manager.save_checkpoint(model, optimizer, total_episodes, wins/total_episodes, filename="model_final.pt")
    
    if prof is not None:
        prof.stop()
        logger.info("Profiler trace saved to ./log/profiler. Run `tensorboard --logdir=./log/profiler` to view.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10, help="Number of episodes to train")
    parser.add_argument('--save_freq', type=int, default=5, help="Save a checkpoint every N episodes")
    parser.add_argument('--profile', action='store_true', help="Enable PyTorch profiler")
    parser.add_argument('--load_bc', action='store_true', help="Initialize model with Behavioral Cloning weights")
    parser.add_argument('--resume', action='store_true', help="Resume from the latest checkpoint")
    args = parser.parse_args()
    
    train(args)
