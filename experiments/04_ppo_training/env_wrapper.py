import sys
import os
import json
import kaggle_environments
import random

_baseline_agent = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../01_baseline/agent")
)
if _baseline_agent not in sys.path:
    sys.path.insert(0, _baseline_agent)

from cg.api import to_observation_class

class PokemonPPOEnv:
    def __init__(self, fixed_opponent=None):
        self.fixed_opponent = fixed_opponent
        self.dragapult_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../03_rule_based_eval/agents/dragapult.py"))
        self.env = kaggle_environments.make("cabt", debug=True)
        self.trainer = None
        self.my_prizes = 6
        self.op_prizes = 6
        
    def get_prizes(self, obs_dict):
        if isinstance(obs_dict, dict) and 'observation' in obs_dict:
            obs_json = obs_dict['observation']
        elif hasattr(obs_dict, 'observation'):
            obs_json = obs_dict.observation
        else:
            obs_json = obs_dict
        
        if isinstance(obs_json, str):
            try:
                obs_json = json.loads(obs_json)
            except Exception:
                return 6, 6

        try:
            obs = to_observation_class(obs_json)
            your_index = obs.current.yourIndex
            my_prizes = sum(1 for p in obs.current.players[your_index].prize if p is not None)
            op_prizes = sum(1 for p in obs.current.players[1 ^ your_index].prize if p is not None)
            return my_prizes, op_prizes
        except Exception:
            return 6, 6

    def reset(self):
        self.my_prizes = 6
        self.op_prizes = 6
        if self.fixed_opponent:
            opponent = self.fixed_opponent
        else:
            opponent = random.choice(["random", self.dragapult_path])
        self.trainer = self.env.train([None, opponent])
        obs = self.trainer.reset()
        if obs:
            self.my_prizes, self.op_prizes = self.get_prizes(obs)
        return obs
        
    def step(self, action):
        obs, reward, done, info = self.trainer.step(action)
        
        my_prizes_now = self.my_prizes
        op_prizes_now = self.op_prizes
        
        if not done and obs:
            my_prizes_now, op_prizes_now = self.get_prizes(obs)
            
        dense_reward = (self.op_prizes - op_prizes_now) - (self.my_prizes - my_prizes_now)
        
        self.my_prizes = my_prizes_now
        self.op_prizes = op_prizes_now
        
        final_reward = dense_reward
        
        if done:
            if reward is not None and reward > 0:
                final_reward += 5.0
            else:
                final_reward -= 5.0
                
        return obs, final_reward, done, info
