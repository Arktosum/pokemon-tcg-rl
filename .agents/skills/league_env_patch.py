import random
import torch
import numpy as np
import os
import sys

from src.env import PTCGEnv
from src.greedy_agent import GreedyAgent
from src.advanced_agents import AdvancedHeuristicAgent
from src.model import PokemonActorCritic

# Import the new KaggleReplayAgent
# Note: Path may need to be adjusted based on PYTHONPATH
try:
    from kaggle_replay_agent import KaggleReplayAgent
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.agents/skills')))
    from kaggle_replay_agent import KaggleReplayAgent


class LeagueEnv(PTCGEnv):
    def __init__(self, ppo_ckpt_path="checkpoints/TOP_ELO_PPO_PEAK.pt"):
        super().__init__()
        self.greedy_agent = GreedyAgent()
        self.advanced_agent = AdvancedHeuristicAgent()
        
        self.past_self = PokemonActorCritic(num_layers=2)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_full_path = os.path.join(base_dir, ppo_ckpt_path)
        
        try:
            checkpoint = torch.load(ckpt_full_path, map_location="cpu", weights_only=True)
            self.past_self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"LeagueEnv Warning: Could not load Past Self ({e}). Using fresh weights.")
        self.past_self.eval()
        
        # Initialize Kaggle Replay Agent
        self.kaggle_agent = KaggleReplayAgent(replays_dir=base_dir)
        
        # Updated opponent pool to natively ingest Kaggle JSON replays
        self.opponent_pool = [
            "RandomAgent",
            "GreedyAgent",
            "AdvancedHeuristicAgent",
            "PastSelf",
            "KaggleReplay"
        ]
        # Adjusted weights for the new pool
        self.weights = [0.05, 0.15, 0.20, 0.30, 0.30]
        
        self.current_opponent_name = None
        self.agent_player_idx = 0
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_opponent_name = random.choices(self.opponent_pool, weights=self.weights, k=1)[0]
        self.agent_player_idx = random.choice([0, 1])
        obs, op_reward, done = self._step_opponent_until_turn(obs)
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
            state_vec = obs["obs"]
            current_player = int(state_vec[2])
            if current_player == self.agent_player_idx:
                break
            action = self._get_opponent_action(obs)
            obs, r, done, _, _ = super().step(action)
            total_op_reward += r
        return obs, total_op_reward, done

    def _get_opponent_action(self, obs):
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        
        if len(valid_actions) == 0:
            return 0
            
        if self.current_opponent_name == "RandomAgent":
            return int(np.random.choice(valid_actions))
        elif self.current_opponent_name == "GreedyAgent":
            return self.greedy_agent.act(obs)
        elif self.current_opponent_name == "AdvancedHeuristicAgent":
            return self.advanced_agent.act(obs, self)
        elif self.current_opponent_name == "PastSelf":
            with torch.no_grad():
                s_tensor = torch.tensor(obs["obs"], dtype=torch.float32).unsqueeze(0)
                m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
                policy, _ = self.past_self(s_tensor, m_tensor)
                p = policy.squeeze(0)[valid_actions]
                if p.sum() > 0:
                    p /= p.sum()
                    return int(np.random.choice(valid_actions, p=p.numpy()))
                else:
                    return int(np.random.choice(valid_actions))
        elif self.current_opponent_name == "KaggleReplay":
            # Delegate to the newly integrated KaggleReplayAgent
            return self.kaggle_agent.act(obs, self)
            
        return int(np.random.choice(valid_actions))
