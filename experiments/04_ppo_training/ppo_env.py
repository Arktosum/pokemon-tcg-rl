import json
import logging
import sys
import os
from typing import List, Any
from kaggle_environments import make

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_baseline", "agent")))
from cg.api import to_observation_class
from parser import get_encoder_input, get_decoder_input

from ppo_types import ParsedObs, StepResult

class PokemonPPOEnv:
    def __init__(self, opponent: str = "random", max_steps: int = 250):
        self.opponent = opponent
        self.max_steps = max_steps
        self.env = make("cabt", configuration={"agent": opponent})
        self.trainer = self.env.train([None, opponent])
        self.current_step = 0
        self.done = True
        self.prev_opp_hp = None

    def _parse_obs(self, raw_obs: Any) -> ParsedObs | None:
        if raw_obs is None: return None
        try:
            obs_json = json.loads(raw_obs) if isinstance(raw_obs, str) else raw_obs
            if isinstance(obs_json, list):
                obs_json = obs_json[0]
            if 'observation' in obs_json: 
                obs_json = obs_json['observation']
            
            obs_class = to_observation_class(obs_json)
            
            try:
                with open(os.path.join(os.path.dirname(__file__), "logs", "action_mask_audit.txt"), "a") as f:
                    f.write(f"--- TURN OPTIONS ---\\n")
                    if obs_class.select and obs_class.select.option:
                        for opt in obs_class.select.option:
                            f.write(f"{str(opt)}\\n")
            except Exception:
                pass

            if obs_class.select is None or obs_class.current is None: return None

            enc_sv = get_encoder_input(obs_class)
            num_options = len(obs_class.select.option) if obs_class.select and obs_class.select.option else 1
            max_count = obs_class.select.maxCount if obs_class.select and hasattr(obs_class.select, 'maxCount') else 1

            actions = [[j] for j in range(num_options)]
            dec_sv = get_decoder_input(obs_class, actions)

            return ParsedObs(
                enc_index=list(enc_sv.index),
                enc_value=list(enc_sv.value),
                enc_offset=list(enc_sv.offset),
                dec_index=list(dec_sv.index),
                dec_value=list(dec_sv.value),
                dec_offset=list(dec_sv.offset),
                num_options=num_options,
                max_count=max_count
            )
        except Exception as e:
            logging.error(f"Error parsing observation: {e}")
            return None

    def reset(self) -> StepResult:
        self.current_step = 0
        obs = self.trainer.reset()
        self.done = False
        self.prev_opp_hp = None
        self.current_prizes = 6
        self.max_concurrent_energies = 0
        parsed = self._parse_obs(obs)
        return StepResult(obs=parsed, reward=0.0, done=self.done, info={})

    def step(self, action: List[int]) -> StepResult:
        self.current_step += 1
        
        # Kaggle environment expects a list of actions or single action?
        # Typically the wrapper passes `action` as is, which needs to be what Kaggle expects.
        try:
            obs, reward, done, info = self.trainer.step(action)
        except Exception as e:
            # Fatal engine crash
            return StepResult(obs=None, reward=-100.0, done=True, info={'is_invalid': True, 'error': str(e)})

        # Kaggle engine reward processing
        raw_reward = reward
        if info is None:
            info = {}

        # Catch Kaggle's literal None reward or an explicit INVALID status
        is_invalid = False
        if done and (raw_reward is None or info.get('status') == 'INVALID' or (isinstance(raw_reward, list) and len(raw_reward) > 0 and raw_reward[0] is None)):
            is_invalid = True
            info['is_invalid'] = True
            info['failed_action'] = action
            reward = -100.0 # Huge penalty for INVALID
        elif raw_reward is None or (isinstance(raw_reward, list) and raw_reward[0] is None):
            reward = -1.0 # Default step penalty if none
        elif isinstance(raw_reward, list):
            reward = float(raw_reward[0])
        else:
            reward = float(raw_reward)

        if self.current_step >= self.max_steps:
            done = True

        self.done = done
        
        try:
            obs_json = json.loads(obs) if isinstance(obs, str) else obs
            if isinstance(obs_json, list):
                obs_json = obs_json[0]
            if 'observation' in obs_json: 
                obs_json = obs_json['observation']
            
            obs_class = to_observation_class(obs_json)
            
            # --- PRIZE CARD REWARD SHAPING ---
            PRIZE_REWARD_WEIGHT = 0.5
            if obs_class.current and obs_class.current.players:
                idx = getattr(obs_class.current, 'yourIndex', 0)
                if 0 <= idx < len(obs_class.current.players):
                    # 1. Prize Cards
                    prizes_after = len(obs_class.current.players[idx].prize)
                    if hasattr(self, 'current_prizes'):
                        prize_delta = self.current_prizes - prizes_after
                        if prize_delta > 0:
                            reward += (prize_delta * PRIZE_REWARD_WEIGHT)
                    self.current_prizes = prizes_after
                    
                    # 2. Energy High-Water Mark
                    current_energies = 0
                    player = obs_class.current.players[idx]
                    if player.active and player.active[0]:
                        current_energies += len(player.active[0].energyCards)
                    if player.bench:
                        for pkmn in player.bench:
                            if pkmn:
                                current_energies += len(pkmn.energyCards)
                    
                    if hasattr(self, 'max_concurrent_energies'):
                        if current_energies > self.max_concurrent_energies:
                            energy_delta = current_energies - self.max_concurrent_energies
                            reward += (energy_delta * 0.05)
                            self.max_concurrent_energies = current_energies
            # ---------------------------------
            
            # --- STEP PENALTY (TIME DECAY) ---
            reward -= 0.001
            # ---------------------------------
            
            current_opp_hp = 0
            if obs_class.opponent and obs_class.opponent.active:
                current_opp_hp += obs_class.opponent.active.hp
            if obs_class.opponent and obs_class.opponent.bench:
                for pkmn in obs_class.opponent.bench:
                    if pkmn: current_opp_hp += pkmn.hp
                    
            if self.prev_opp_hp is not None:
                hp_diff = self.prev_opp_hp - current_opp_hp
                if hp_diff > 0:
                    reward += (hp_diff * 0.001)
                    
            self.prev_opp_hp = current_opp_hp
        except Exception:
            pass
            
        parsed = self._parse_obs(obs)
        return StepResult(obs=parsed, reward=reward, done=done, info=info)
