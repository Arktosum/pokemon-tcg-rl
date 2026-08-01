import os
import sys
import numpy as np
import torch
from kaggle_environments import make

# Ensure baseline is in path so we can import agent modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline')))
from agent.parser import get_encoder_input, get_decoder_input
from agent.cg.api import to_observation_class

class PokemonEnvWrapper:
    def __init__(self, opponent_func):
        self.env = make("cabt", debug=False)
        self.opponent_func = opponent_func
        self.trainer = None
        # HP tracking for dense reward (prize zone is all-None face-down → unusable)
        self.opp_active_hp = None
        self.opp_active_max_hp = None
        self.my_active_hp = None
        self.my_active_max_hp = None
        self.current_obs = None
        self.deck_csv = self._read_deck()

    def _read_deck(self) -> list[int]:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent', 'deck.csv'))
        if not os.path.exists(file_path):
            file_path = "/kaggle_simulations/agent/deck.csv"
        try:
            with open(file_path, "r") as file:
                csv = file.read().split("\n")
            deck = []
            for i in range(60):
                if csv[i].strip():
                    deck.append(int(csv[i]))
            return deck
        except Exception:
            return [1] * 60

    def reset(self, opponent_func_override=None):
        # Allow dynamic opponent swapping
        if opponent_func_override is not None:
            self.opponent_func = opponent_func_override
            
        # Re-initialize trainer to reset internal state completely
        self.trainer = self.env.train([None, self.opponent_func])
        raw_obs = self.trainer.reset()
        
        # Reset HP trackers
        self.opp_active_hp = None
        self.opp_active_max_hp = None
        self.my_active_hp = None
        self.my_active_max_hp = None
        self.current_obs = raw_obs
        
        # Kaggle env requires deck submission if select is None
        obs_obj = to_observation_class(raw_obs)
        if obs_obj.select is None:
            raw_obs, r, d, i = self.trainer.step(self.deck_csv)
            self.current_obs = raw_obs
            
        return self._encode_obs(self.current_obs)

    def _encode_obs(self, raw_obs):
        obs = to_observation_class(raw_obs)
        if obs.select is None:
            return None
            
        sv_enc = get_encoder_input(obs, self.deck_csv)
        
        legal_count = len(obs.select.option)
        actions = [[i] for i in range(legal_count)]
        sv_dec = get_decoder_input(obs, actions)
        
        return {
            "enc_indices": np.array(sv_enc.index, dtype=np.int32),
            "enc_weights": np.array(sv_enc.value, dtype=np.float32),
            "enc_offsets": np.array(sv_enc.offset, dtype=np.int32),
            "dec_indices": np.array(sv_dec.index, dtype=np.int32),
            "dec_weights": np.array(sv_dec.value, dtype=np.float32),
            "dec_offsets": np.array(sv_dec.offset, dtype=np.int32),
            "legal_count": legal_count
        }

    def _read_hp(self, raw_obs):
        """Extract my and opponent active HP from obs. Returns (my_hp, my_max, opp_hp, opp_max) or Nones."""
        try:
            obs = to_observation_class(raw_obs)
            if obs.current is None or not obs.current.players:
                return None, None, None, None
            my_idx = obs.current.yourIndex
            opp_idx = 1 - my_idx
            my_p = obs.current.players[my_idx]
            opp_p = obs.current.players[opp_idx]
            
            my_hp = my_p.active[0].hp if my_p.active and my_p.active[0] else None
            my_max = my_p.active[0].maxHp if my_p.active and my_p.active[0] else None
            opp_hp = opp_p.active[0].hp if opp_p.active and opp_p.active[0] else None
            opp_max = opp_p.active[0].maxHp if opp_p.active and opp_p.active[0] else None
            return my_hp, my_max, opp_hp, opp_max
        except Exception:
            return None, None, None, None

    def step(self, action_idx: int):
        raw_obs, engine_reward, done, info = self.trainer.step([action_idx])
        self.current_obs = raw_obs
        
        reward = self._shape_reward(engine_reward, done, raw_obs)
        
        encoded_obs = None
        if not done:
            encoded_obs = self._encode_obs(raw_obs)
            
        info = info or {}
        info["engine_reward"] = engine_reward
            
        return encoded_obs, reward, done, info

    def _shape_reward(self, engine_reward, done, raw_obs):
        r = 0.0
        
        if done:
            status = self.env.state[0].status if self.env.state else ""
            if status in ["ERROR", "INVALID"]:
                r -= 1.0
            elif engine_reward == 1:
                r += 1.0
            elif engine_reward == -1:
                r -= 1.0
            else:
                r -= 0.1  # Draw penalty
        else:
            # Dense reward: HP-based progress (prize zone is all-None face-down, unusable)
            my_hp, my_max, opp_hp, opp_max = self._read_hp(raw_obs)
            
            if opp_hp is not None and self.opp_active_max_hp is not None:
                prev_opp_hp = self.opp_active_hp if self.opp_active_hp is not None else opp_max
                if opp_max and opp_max > 0:
                    dmg_dealt = max(0, prev_opp_hp - opp_hp)
                    r += 0.1 * (dmg_dealt / opp_max)  # up to +0.1 for a full KO
            
            if my_hp is not None and self.my_active_max_hp is not None:
                prev_my_hp = self.my_active_hp if self.my_active_hp is not None else my_max
                if my_max and my_max > 0:
                    dmg_taken = max(0, prev_my_hp - my_hp)
                    r -= 0.05 * (dmg_taken / my_max)  # up to -0.05 for full KO on us
            
            # Update HP trackers
            self.opp_active_hp = opp_hp
            self.opp_active_max_hp = opp_max
            self.my_active_hp = my_hp
            self.my_active_max_hp = my_max
                
        return r

if __name__ == "__main__":
    # Smoke test
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))
    from greedy_agent import greedy_agent
    
    env = PokemonEnvWrapper(greedy_agent)
    obs = env.reset()
    print("Env reset successfully. Legal moves:", obs["legal_count"])
    obs, reward, done, info = env.step(0)
    print("Env stepped successfully. Done:", done, "Reward:", reward)

