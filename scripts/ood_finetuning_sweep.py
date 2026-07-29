"""
Phase 82: Patched Replay Agent + 500-Episode Convergence Sweep

Key finding from replay audit:
- Each episode step's `action` field is a LIST of card IDs for the deck-selection phase,
  but after deck selection, action[0] is the INDEX into select.option[] for that turn.
- The opponent (Agent 0) makes most moves; our agent (Agent 1) only acts on steps where
  status == 'ACTIVE'.
- The replay agent uses this index directly as the env action, enabling faithful OOD replay.

Convergence criteria:
- TRUE convergence: entropy downward trend (final 100ep avg < initial 100ep avg by >15%)
  AND avg reward trend toward positive.
- Catastrophic forgetting: entropy collapses below 0.3 AND reward also drops sharply.
"""

import os
import sys
import gc
import glob
import json
import random
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
from league_env import LeagueEnv
from ppo_buffer import PPOBuffer


# -----------------------------------------------------------------------
# TRUE KaggleReplayAgent: maps env step counter -> recorded action index
# -----------------------------------------------------------------------
class KaggleReplayAgent:
    def __init__(self, replays_dir=ROOT, agent_id=0):
        """
        agent_id: which agent in the replay this bot mimics (0 or 1).
        We use agent_id=0 (the opponent in our lost match) as the OOD opponent.
        """
        self.agent_id = agent_id
        self.all_sequences = []   # list of lists: [(action_index, n_options), ...]

        for filepath in glob.glob(os.path.join(replays_dir, "*-replay.json")):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                seq = self._extract_sequence(data)
                if seq:
                    self.all_sequences.append(seq)
                    print(f"  Loaded {len(seq)} action steps from {os.path.basename(filepath)}")
            except Exception as e:
                sys.stderr.write(f"  [REPLAY LOAD ERROR] {filepath}: {e}\n")

        # State for current episode
        self._current_seq = []
        self._seq_ptr = 0

    def _extract_sequence(self, data):
        """Extract (action_index, n_options) pairs for self.agent_id from replay."""
        seq = []
        steps = data.get("steps", [])
        for step in steps:
            if not isinstance(step, list) or len(step) <= self.agent_id:
                continue
            ag = step[self.agent_id]
            if not ag:
                continue
            if ag.get("status") != "ACTIVE":
                continue
            action = ag.get("action")
            if not action or len(action) == 0:
                continue
            obs = ag.get("observation", {})
            sel = obs.get("select")
            if not isinstance(sel, dict):
                continue
            n_opts = len(sel.get("option", []))
            # action[0] is index into option[] for post-deck-selection steps
            # For deck selection (step 1), action is a long list of card IDs — skip it
            action_val = action[0]
            if action_val < n_opts:
                seq.append((action_val, n_opts))
            # else it's a card-ID format (deck selection) — skip
        return seq

    def reset_episode(self):
        """Call at the start of each episode to pick a random replay sequence."""
        if self.all_sequences:
            self._current_seq = random.choice(self.all_sequences)
        else:
            self._current_seq = []
        self._seq_ptr = 0

    def act(self, obs):
        mask = obs["action_mask"]
        valid_actions = np.where(np.array(mask) == 1)[0]
        n_valid = len(valid_actions)

        if n_valid == 0:
            return 0

        if self._seq_ptr < len(self._current_seq):
            rec_action_idx, rec_n_opts = self._current_seq[self._seq_ptr]
            self._seq_ptr += 1

            # Synchrony check: does the recorded n_options match current branching factor?
            if rec_n_opts == n_valid:
                # Perfect match — replay the exact action index
                return int(valid_actions[rec_action_idx])
            else:
                # State desync: branching factor doesn't match
                sys.stderr.write(
                    f"[REPLAY DESYNC] step={self._seq_ptr-1}: "
                    f"recorded n_opts={rec_n_opts}, live n_valid={n_valid}. "
                    f"Falling back to random.\n"
                )
                return int(np.random.choice(valid_actions))
        else:
            # Ran out of replay data — fallback
            return int(np.random.choice(valid_actions))


# -----------------------------------------------------------------------
# Monkey-patch LeagueEnv to add KaggleReplayAgent into pool
# -----------------------------------------------------------------------
def patch_league_env(env_cls, replay_agent):
    orig_init   = env_cls.__init__
    orig_reset  = env_cls.reset
    orig_get_op = env_cls._get_opponent_action

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.kaggle_replay_agent = replay_agent
        if "KaggleReplay" not in self.opponent_pool:
            self.opponent_pool.append("KaggleReplay")
            # Add weight equal to current last weight, then renormalize
            self.weights.append(self.weights[-1])
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def new_reset(self, seed=None, options=None):
        self.kaggle_replay_agent.reset_episode()
        return orig_reset(self, seed=seed, options=options)

    def new_get_op(self, obs):
        if self.current_opponent_name == "KaggleReplay":
            return self.kaggle_replay_agent.act(obs)
        return orig_get_op(self, obs)

    env_cls.__init__           = new_init
    env_cls.reset              = new_reset
    env_cls._get_opponent_action = new_get_op
    return env_cls


# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------
def run_sweep(num_episodes=500, report_interval=50):
    print("=" * 70)
    print("PHASE 82: PATCHED REPLAY AGENT + 500-EPISODE CONVERGENCE SWEEP")
    print("=" * 70)

    # Load BC model
    print("\nLoading TOP_ELO_BC_MODEL_FINAL.pt (num_layers=3)...")
    model = PokemonActorCritic(num_layers=3)
    ckpt_path = os.path.join(ROOT, "checkpoints", "TOP_ELO_BC_MODEL_FINAL.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.train()
    print("Model loaded successfully.")

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    buffer    = PPOBuffer()

    clip_param   = 0.1
    entropy_coef = 0.02
    value_coef   = 0.5
    ppo_epochs   = 4

    # Setup patched env
    print("\nInitializing patched LeagueEnv + KaggleReplayAgent(agent_id=0)...")
    replay_agent = KaggleReplayAgent(replays_dir=ROOT, agent_id=0)
    patch_league_env(LeagueEnv, replay_agent)
    env = LeagueEnv()
    print(f"Opponent pool: {env.opponent_pool}")
    print(f"Weights:       {[f'{w:.3f}' for w in env.weights]}")

    print("\nReporting at 50-episode intervals:")
    print("-" * 70)
    print(f"{'Episode':>8} | {'Avg Reward':>10} | {'Policy Entropy':>14} | {'Value Loss':>10}")
    print("-" * 70)

    episode_rewards    = []
    window_entropy     = []
    window_val_loss    = []
    last_entropy  = 0.0
    last_val_loss = 0.0

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < 200:
            state_vec = obs["obs"]
            mask      = obs["action_mask"]

            s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            m_tensor = torch.tensor(mask,      dtype=torch.int8).unsqueeze(0)

            with torch.no_grad():
                policy, value = model(s_tensor, m_tensor)

            p = policy.squeeze(0)
            valid_actions = np.where(mask == 1)[0]
            if len(valid_actions) == 0:
                break

            p_valid = p[valid_actions]
            p_sum   = p_valid.sum()
            if p_sum > 1e-8:
                p_valid = p_valid / p_sum
            else:
                p_valid = torch.ones(len(valid_actions)) / len(valid_actions)

            dist       = torch.distributions.Categorical(p_valid)
            action_idx = dist.sample()
            action     = int(valid_actions[action_idx.item()])
            log_prob   = dist.log_prob(action_idx).item()
            val_item   = value.item()

            try:
                obs, reward, done, _, _ = env.step(action)
            except Exception:
                break

            ep_reward += reward
            buffer.store(state_vec, mask, action, reward, val_item, log_prob, 1 - int(done))
            step += 1

        episode_rewards.append(ep_reward)

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
                policy, values = model(b_states, b_masks)
                values = values.squeeze(1)

                clamped = torch.clamp(policy, 1e-8, 1.0)
                valid_p = clamped * b_masks
                valid_p = valid_p / (valid_p.sum(dim=1, keepdim=True) + 1e-8)
                log_vp  = torch.log(valid_p + 1e-8)
                entropy = -torch.sum(valid_p * log_vp * b_masks, dim=1).mean()

                dist        = torch.distributions.Categorical(clamped)
                new_lp      = dist.log_prob(b_actions)
                ratio       = torch.exp(new_lp - b_old_lp)
                surr1       = ratio * b_adv
                surr2       = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * b_adv
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, b_returns)
                loss        = actor_loss + value_coef * critic_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            last_entropy  = entropy.item()
            last_val_loss = critic_loss.item()
            buffer.clear()
            gc.collect()

        window_entropy.append(last_entropy)
        window_val_loss.append(last_val_loss)

        if episode % report_interval == 0:
            avg_reward    = np.mean(episode_rewards[-report_interval:])
            avg_entropy   = np.mean(window_entropy[-report_interval:])
            avg_val_loss  = np.mean(window_val_loss[-report_interval:])
            print(f"{episode:>8} | {avg_reward:>10.4f} | {avg_entropy:>14.4f} | {avg_val_loss:>10.4f}", flush=True)

    # Save
    out_path = os.path.join(ROOT, "checkpoints", "BC_CONVERGENCE_SWEEP.pt")
    torch.save({"model_state_dict": model.state_dict()}, out_path)

    # Convergence analysis
    first_100_entropy = np.mean(window_entropy[:100])
    last_100_entropy  = np.mean(window_entropy[-100:])
    pct_drop = (first_100_entropy - last_100_entropy) / (first_100_entropy + 1e-8) * 100
    first_100_reward  = np.mean(episode_rewards[:100])
    last_100_reward   = np.mean(episode_rewards[-100:])

    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS:")
    print(f"  Entropy ep1-100 avg:   {first_100_entropy:.4f}")
    print(f"  Entropy ep401-500 avg: {last_100_entropy:.4f}")
    print(f"  Entropy drop:          {pct_drop:.1f}%")
    print(f"  Reward ep1-100 avg:    {first_100_reward:.4f}")
    print(f"  Reward ep401-500 avg:  {last_100_reward:.4f}")
    verdict = "CONVERGED" if pct_drop > 15 else "NOT CONVERGED"
    print(f"  Verdict:               {verdict}")
    print(f"\nCheckpoint saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_sweep(num_episodes=500, report_interval=50)
