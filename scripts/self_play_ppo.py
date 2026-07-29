"""
Phase 84: Self-Play PPO Ascension
- 100% PastSelf opponent pool, win-rate gating
- Resumable from any periodic or gate checkpoint
- Per-report timing, extended metrics (avg steps/ep, wins, win-streak)
- Periodic checkpoints every --save_every N episodes
"""

import os
import sys
import gc
import copy
import time
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENGINE_PATH = os.path.join(ROOT, 'data', 'sample_submission', 'sample_submission')
for p in [ROOT, os.path.join(ROOT, 'src'), ENGINE_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import PokemonActorCritic
from env import PTCGEnv
from ppo_buffer import PPOBuffer


# -----------------------------------------------------------------------
# Self-Play Env
# -----------------------------------------------------------------------
class SelfPlayEnv(PTCGEnv):
    def __init__(self):
        super().__init__()
        self.past_self_model = None
        self.agent_player_idx = 0

    def set_past_self(self, state_dict):
        ps = PokemonActorCritic(num_layers=3)
        ps.load_state_dict(copy.deepcopy(state_dict))
        ps.eval()
        self.past_self_model = ps

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        obs, _, _ = self._step_opponent_until_turn(obs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if not done:
            obs, op_reward, done = self._step_opponent_until_turn(obs)
            reward -= op_reward
        return obs, reward, done, truncated, info

    def _step_opponent_until_turn(self, obs):
        total_op_reward = 0.0
        done = self.is_done
        while not done:
            current_player = int(obs["obs"][2])
            if current_player == self.agent_player_idx:
                break
            action = self._past_self_act(obs)
            obs, r, done, _, _ = super().step(action)
            total_op_reward += r
        return obs, total_op_reward, done

    def _past_self_act(self, obs):
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            return 0
        if self.past_self_model is None:
            return int(np.random.choice(valid_actions))
        with torch.no_grad():
            s = torch.tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
            policy, _ = self.past_self_model(s, m)
        p = policy.squeeze(0)[valid_actions]
        p_sum = p.sum()
        if p_sum > 1e-8:
            p = p / p_sum
            return int(np.random.choice(valid_actions, p=p.numpy()))
        return int(np.random.choice(valid_actions))


# -----------------------------------------------------------------------
# Win-rate evaluation
# -----------------------------------------------------------------------
def evaluate_win_rate(model, past_self_sd, n_games=100):
    env = SelfPlayEnv()
    env.set_past_self(past_self_sd)
    model.eval()
    wins, total_steps = 0, 0
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        steps = 0
        last_reward = 0
        while not done and steps < 200:
            mask = obs["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) == 0:
                break
            with torch.no_grad():
                s = torch.tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
                policy, _ = model(s, m)
            p = policy.squeeze(0)[valid_actions]
            p_sum = p.sum()
            if p_sum > 1e-8:
                p = p / p_sum
                action = int(np.random.choice(valid_actions, p=p.numpy()))
            else:
                action = int(np.random.choice(valid_actions))
            obs, last_reward, done, _, _ = env.step(action)
            steps += 1
        total_steps += steps
        if last_reward > 0:
            wins += 1
    model.train()
    return wins / n_games, total_steps / n_games


# -----------------------------------------------------------------------
# Checkpoint save/load
# -----------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, episode, past_self_sd,
                    past_self_update_count, episode_rewards, meta):
    torch.save({
        "model_state_dict":          model.state_dict(),
        "optimizer_state_dict":      optimizer.state_dict(),
        "episode":                   episode,
        "past_self_state_dict":      past_self_sd,
        "past_self_update_count":    past_self_update_count,
        "episode_rewards":           episode_rewards[-2000:],  # keep last 2000
        "meta":                      meta,
    }, path)
    print(f"[CKPT] Saved: {os.path.basename(path)}", flush=True)


def load_checkpoint(path, model, optimizer):
    print(f"[RESUME] Loading checkpoint: {path}", flush=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_episode         = ckpt["episode"] + 1
    past_self_sd          = ckpt["past_self_state_dict"]
    past_self_update_count = ckpt["past_self_update_count"]
    episode_rewards       = ckpt.get("episode_rewards", [])
    meta                  = ckpt.get("meta", {})
    print(f"[RESUME] Resuming from episode {start_episode}, "
          f"PastSelf updates so far: {past_self_update_count}", flush=True)
    return start_episode, past_self_sd, past_self_update_count, episode_rewards, meta


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def run_self_play(args):
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 75)
    print("PHASE 84: SELF-PLAY PPO ASCENSION")
    print("=" * 75)
    print(f"  Total episodes:        {args.episodes}")
    print(f"  Report interval:       every {args.report_every} episodes")
    print(f"  Save every:            every {args.save_every} episodes")
    print(f"  Eval interval:         every {args.eval_every} episodes")
    print(f"  Eval games per gate:   {args.eval_games}")
    print(f"  Win-rate gate:         > {args.win_rate*100:.0f}%")
    print(f"  Learning rate:         {args.lr}")
    print(f"  Entropy coef:          {args.entropy_coef}")
    print(f"  Resume from:           {args.resume or 'None (fresh start)'}")
    print("=" * 75)

    # -- Model + Optimizer --
    model = PokemonActorCritic(num_layers=3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_episode          = 1
    past_self_update_count = 0
    episode_rewards        = []
    all_episode_steps      = []
    cumulative_wins        = 0
    meta                   = {"gate_history": []}

    # -- Resume or fresh load --
    if args.resume and os.path.exists(args.resume):
        start_episode, past_self_sd, past_self_update_count, episode_rewards, meta = \
            load_checkpoint(args.resume, model, optimizer)
        all_episode_steps = []
    else:
        baseline = args.baseline or os.path.join(ckpt_dir, "BC_CONVERGENCE_SWEEP.pt")
        print(f"\nLoading baseline: {os.path.basename(baseline)} ...", flush=True)
        raw = torch.load(baseline, map_location="cpu", weights_only=True)
        model.load_state_dict(raw.get("model_state_dict", raw))
        past_self_sd = copy.deepcopy(model.state_dict())
        print("Baseline loaded.\n", flush=True)

    model.train()
    env = SelfPlayEnv()
    env.set_past_self(past_self_sd)

    buffer = PPOBuffer()
    clip_param  = 0.2
    value_coef  = 0.5
    ppo_epochs  = 4

    # -- Header --
    print("-" * 95)
    print(f"{'Episode':>8} | {'AvgReward':>9} | {'Entropy':>7} | {'ValLoss':>7} | "
          f"{'AvgSteps':>8} | {'WinRate%':>8} | {'PSupd':>5} | {'Time/100ep':>10}")
    print("-" * 95)

    last_entropy     = 0.0
    last_val_loss    = 0.0
    window_start_t   = time.time()
    total_start_t    = time.time()
    win_streak       = 0   # consecutive gate passes

    for episode in range(start_episode, args.episodes + 1):

        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done and ep_steps < 200:
            mask = obs["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) == 0:
                break

            s = torch.tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(mask,       dtype=torch.int8).unsqueeze(0)
            with torch.no_grad():
                policy, value = model(s, m)

            p = policy.squeeze(0)[valid_actions]
            p_sum = p.sum()
            p = (p / p_sum) if p_sum > 1e-8 else torch.ones(len(valid_actions)) / len(valid_actions)

            dist       = torch.distributions.Categorical(p)
            action_idx = dist.sample()
            action     = int(valid_actions[action_idx.item()])
            log_prob   = dist.log_prob(action_idx).item()
            val_item   = value.item()

            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            buffer.store(obs["obs"], mask, action, reward, val_item, log_prob, 1 - int(done))
            ep_steps += 1

        episode_rewards.append(ep_reward)
        all_episode_steps.append(ep_steps)
        if ep_reward > 0:
            cumulative_wins += 1

        # -- PPO update --
        if len(buffer.rewards) > 0:
            advantages, returns = buffer.compute_gae(0.0)
            b_states  = torch.tensor(np.array(buffer.states),       dtype=torch.float32)
            b_masks   = torch.tensor(np.array(buffer.action_masks), dtype=torch.float32)
            b_actions = torch.tensor(buffer.actions,                 dtype=torch.long)
            b_old_lp  = torch.tensor(buffer.log_probs,              dtype=torch.float32)
            b_returns = torch.tensor(returns,                        dtype=torch.float32)
            b_adv     = torch.tensor(advantages,                     dtype=torch.float32)
            if len(b_adv) > 1:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            for _ in range(ppo_epochs):
                pol, vals = model(b_states, b_masks)
                vals = vals.squeeze(1)
                clamped = torch.clamp(pol, 1e-8, 1.0)
                vp      = clamped * b_masks
                vp      = vp / (vp.sum(dim=1, keepdim=True) + 1e-8)
                entropy = -torch.sum(vp * torch.log(vp + 1e-8) * b_masks, dim=1).mean()
                dist_t     = torch.distributions.Categorical(clamped)
                ratio      = torch.exp(dist_t.log_prob(b_actions) - b_old_lp)
                surr       = torch.min(ratio * b_adv,
                                       torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * b_adv)
                actor_loss  = -surr.mean()
                critic_loss = F.mse_loss(vals, b_returns)
                loss        = actor_loss + value_coef * critic_loss - args.entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            last_entropy  = entropy.item()
            last_val_loss = critic_loss.item()
            buffer.clear()
            gc.collect()

        # -- Report --
        if episode % args.report_every == 0:
            n = args.report_every
            window_rewards = episode_rewards[-n:]
            window_steps   = all_episode_steps[-n:]
            avg_reward  = np.mean(window_rewards)
            avg_steps   = np.mean(window_steps)
            wins_in_win = sum(1 for r in window_rewards if r > 0)
            win_rate_w  = wins_in_win / len(window_rewards) * 100
            elapsed     = time.time() - window_start_t
            print(f"{episode:>8} | {avg_reward:>9.4f} | {last_entropy:>7.4f} | {last_val_loss:>7.4f} | "
                  f"{avg_steps:>8.1f} | {win_rate_w:>8.1f} | {past_self_update_count:>5} | {elapsed:>8.1f}s",
                  flush=True)
            window_start_t = time.time()

        # -- Periodic checkpoint --
        if episode % args.save_every == 0:
            path = os.path.join(ckpt_dir, f"SELF_PLAY_PERIODIC_ep{episode:06d}.pt")
            save_checkpoint(path, model, optimizer, episode, past_self_sd,
                            past_self_update_count, episode_rewards, meta)

        # -- Win-rate gate eval --
        if episode % args.eval_every == 0:
            gate_t0 = time.time()
            print(f"\n[EVAL ep={episode}] Running {args.eval_games} games vs PastSelf ...", flush=True)
            wr, avg_eval_steps = evaluate_win_rate(model, past_self_sd, n_games=args.eval_games)
            gate_elapsed = time.time() - gate_t0
            print(f"[EVAL ep={episode}] WinRate={wr*100:.1f}% | AvgSteps={avg_eval_steps:.1f} | "
                  f"Threshold={args.win_rate*100:.0f}% | EvalTime={gate_elapsed:.1f}s", flush=True)

            if wr > args.win_rate:
                win_streak += 1
                past_self_sd = copy.deepcopy(model.state_dict())
                past_self_update_count += 1
                env.set_past_self(past_self_sd)
                gate_path = os.path.join(ckpt_dir,
                    f"SELF_PLAY_GATE_{past_self_update_count:03d}_ep{episode:06d}.pt")
                save_checkpoint(gate_path, model, optimizer, episode, past_self_sd,
                                past_self_update_count, episode_rewards, meta)
                meta["gate_history"].append({"episode": episode, "win_rate": wr,
                                             "update_num": past_self_update_count})
                print(f"[GATE PASSED [OK]] Update #{past_self_update_count} | "
                      f"Streak={win_streak} | {os.path.basename(gate_path)}", flush=True)
            else:
                win_streak = 0
                print(f"[GATE FAILED [FAIL]] {wr*100:.1f}% < {args.win_rate*100:.0f}%. "
                      f"PastSelf unchanged.", flush=True)
            print(flush=True)

    # -- Final --
    final_path = os.path.join(ckpt_dir, "SELF_PLAY_FINAL.pt")
    save_checkpoint(final_path, model, optimizer, args.episodes, past_self_sd,
                    past_self_update_count, episode_rewards, meta)
    total_elapsed = time.time() - total_start_t
    print(f"\n{'='*75}")
    print(f"SELF-PLAY COMPLETE | {args.episodes} episodes | "
          f"PastSelf updates: {past_self_update_count} | Total time: {total_elapsed/3600:.2f}h")
    print(f"Final checkpoint: {final_path}")
    print(f"{'='*75}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Play PPO Ascension -- Phase 84")
    parser.add_argument("--episodes",      type=int,   default=10000,  help="Total training episodes")
    parser.add_argument("--report_every",  type=int,   default=100,    help="Print metrics every N episodes")
    parser.add_argument("--save_every",    type=int,   default=1000,   help="Save periodic checkpoint every N episodes")
    parser.add_argument("--eval_every",    type=int,   default=500,    help="Run win-rate gate eval every N episodes")
    parser.add_argument("--eval_games",    type=int,   default=100,    help="Games per gate evaluation")
    parser.add_argument("--win_rate",      type=float, default=0.55,   help="Win-rate threshold to update PastSelf")
    parser.add_argument("--lr",            type=float, default=1e-5,   help="Learning rate")
    parser.add_argument("--entropy_coef", type=float,  default=0.05,   help="Entropy regularisation coefficient")
    parser.add_argument("--baseline",      type=str,   default=None,   help="Path to starting checkpoint (fresh start)")
    parser.add_argument("--resume",        type=str,   default=None,   help="Path to periodic/gate checkpoint to resume from")
    args = parser.parse_args()
    run_self_play(args)
