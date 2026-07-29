import json
import random
import numpy as np

class KaggleReplayOpponent:
    def __init__(self, replay_dir="replays/"):
        self.replay_dir = replay_dir
        self.replays = []
        # In a real scenario, we would load all json files from the directory
        
    def act(self, obs, env):
        # A mock implementation for ingesting Kaggle JSON replays
        # In actual usage, this would parse the matching state and return the recorded action
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) > 0:
            return int(np.random.choice(valid_actions))
        return 0

def upgrade_league_env_pool(league_env_class):
    """
    Upgrades the LeagueEnv to include KaggleReplayOpponent in its pool.
    """
    original_init = league_env_class.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.replay_agent = KaggleReplayOpponent()
        self.opponent_pool.append("KaggleReplayOpponent")
        
        # Adjust weights to accommodate the new opponent
        # Original: [0.10, 0.20, 0.30, 0.40]
        self.weights = [0.10, 0.20, 0.25, 0.25, 0.20] # Added Replay agent weight
        
    league_env_class.__init__ = new_init
    
    original_get_action = league_env_class._get_opponent_action
    def new_get_action(self, obs):
        if self.current_opponent_name == "KaggleReplayOpponent":
            return self.replay_agent.act(obs, self)
        return original_get_action(self, obs)
        
    league_env_class._get_opponent_action = new_get_action
    
    return league_env_class
