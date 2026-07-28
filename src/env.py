import sys
import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add Kaggle simulation engine to path
ENGINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_submission", "sample_submission")
if ENGINE_PATH not in sys.path:
    sys.path.append(ENGINE_PATH)

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, Observation

MAX_OPTIONS = 500
MAX_BENCH = 5

class PTCGEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # V2: Increased size to 120, max val 3000 for Card IDs
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=3000, shape=(120,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(MAX_OPTIONS,), dtype=np.int8)
        })
        self.action_space = spaces.Discrete(MAX_OPTIONS)
        self.deck0 = None
        self.deck1 = None
        self.is_done = False

    def _read_deck(self):
        deck_path = os.path.join(ENGINE_PATH, "deck.csv")
        with open(deck_path, "r") as f:
            return [int(x) for x in f.read().strip().split("\n")[:60]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.deck0 is None:
            self.deck0 = self._read_deck()
            self.deck1 = self._read_deck()
        
        self.max_bench_size = [0, 0]
        self.max_total_energy = [0, 0]
        
        obs_dict, _ = battle_start(self.deck0, self.deck1)
        self.is_done = False
        return self._process_obs(obs_dict), {}

    def _process_obs(self, obs_dict):
        obs = to_observation_class(obs_dict)
        state_vec = np.zeros(120, dtype=np.float32)
        mask = np.zeros(MAX_OPTIONS, dtype=np.int8)
        
        self.current_obs = obs
        
        if obs.select is not None:
            valid_len = min(len(obs.select.option), MAX_OPTIONS)
            mask[:valid_len] = 1
            
        if obs.current is not None:
            state = obs.current
            # Global Game State
            state_vec[0] = state.turn
            state_vec[1] = state.turnActionCount
            state_vec[2] = state.yourIndex
            state_vec[3] = state.firstPlayer
            
            idx = 4
            for p_idx in [0, 1]:
                p_state = state.players[p_idx]
                
                # Handcrafted features
                state_vec[idx] = p_state.deckCount / 60.0; idx += 1
                state_vec[idx] = p_state.handCount / 60.0; idx += 1
                state_vec[idx] = p_state.benchMax / 5.0; idx += 1
                state_vec[idx] = len(p_state.prize) / 6.0; idx += 1
                state_vec[idx] = float(p_state.poisoned); idx += 1
                state_vec[idx] = float(p_state.burned); idx += 1
                state_vec[idx] = float(p_state.asleep); idx += 1
                state_vec[idx] = float(p_state.paralyzed); idx += 1
                state_vec[idx] = float(p_state.confused); idx += 1
                
                # Active Pokemon (Card ID, HP, MaxHP, Energy Count)
                if p_state.active and p_state.active[0]:
                    pk = p_state.active[0]
                    state_vec[idx] = pk.id; idx += 1
                    state_vec[idx] = pk.hp / 350.0; idx += 1
                    state_vec[idx] = pk.maxHp / 350.0; idx += 1
                    state_vec[idx] = len(pk.energies) / 10.0; idx += 1
                else:
                    idx += 4
                    
                # Bench Pokemon (Pad to 5)
                for b_i in range(5):
                    if b_i < len(p_state.bench):
                        pk = p_state.bench[b_i]
                        state_vec[idx] = pk.id; idx += 1
                        state_vec[idx] = pk.hp / 350.0; idx += 1
                        state_vec[idx] = pk.maxHp / 350.0; idx += 1
                        state_vec[idx] = len(pk.energies) / 10.0; idx += 1
                    else:
                        idx += 4
            
            # Check win condition
            if state.result != -1:
                self.is_done = True
                self.winner = state.result
                
        return {"obs": state_vec, "action_mask": mask}

    def step(self, action):
        if self.is_done:
            return {"obs": np.zeros(120, dtype=np.float32), "action_mask": np.zeros(MAX_OPTIONS, dtype=np.int8)}, 0.0, True, False, {}
            
        obs = self.current_obs
        select_list = []
        acting_player = obs.current.yourIndex
        
        # Save prize counts before stepping
        p0_prizes_before = len(obs.current.players[0].prize)
        p1_prizes_before = len(obs.current.players[1].prize)
        
        if obs.select is not None:
            opts = obs.select.option
            num_opts = len(opts)
            
            if action < num_opts:
                select_list.append(int(action))
            else:
                select_list.append(random.randint(0, num_opts - 1))
                
            min_c = obs.select.minCount
            available = list(range(num_opts))
            if select_list[0] in available:
                available.remove(select_list[0])
                
            while len(select_list) < min_c and available:
                nxt = random.choice(available)
                select_list.append(nxt)
                available.remove(nxt)
                
        try:
            obs_dict = battle_select(select_list)
            next_state = self._process_obs(obs_dict)
        except Exception as e:
            self.is_done = True
            return {"obs": np.zeros(120, dtype=np.float32), "action_mask": np.zeros(MAX_OPTIONS, dtype=np.int8)}, -1.0, True, False, {}

        reward = 0.0
        if self.is_done:
            if self.winner == acting_player:
                reward = 1.0
            elif self.winner == 1 - acting_player:
                reward = -1.0
            else:
                reward = 0.0
        else:
            p0_prizes_after = len(self.current_obs.current.players[0].prize)
            p1_prizes_after = len(self.current_obs.current.players[1].prize)
            p0_taken = p0_prizes_before - p0_prizes_after
            p1_taken = p1_prizes_before - p1_prizes_after
            
            p0_bench_after = len(self.current_obs.current.players[0].bench)
            p1_bench_after = len(self.current_obs.current.players[1].bench)
            
            p0_energy_after = 0
            if self.current_obs.current.players[0].active and self.current_obs.current.players[0].active[0]:
                p0_energy_after += len(self.current_obs.current.players[0].active[0].energies)
            for pk in self.current_obs.current.players[0].bench:
                p0_energy_after += len(pk.energies)
                
            p1_energy_after = 0
            if self.current_obs.current.players[1].active and self.current_obs.current.players[1].active[0]:
                p1_energy_after += len(self.current_obs.current.players[1].active[0].energies)
            for pk in self.current_obs.current.players[1].bench:
                p1_energy_after += len(pk.energies)
                
            curr_reward = 0.0
            if p0_bench_after > self.max_bench_size[0]:
                if acting_player == 0: curr_reward += (p0_bench_after - self.max_bench_size[0]) * 0.05
                self.max_bench_size[0] = p0_bench_after
                
            if p1_bench_after > self.max_bench_size[1]:
                if acting_player == 1: curr_reward += (p1_bench_after - self.max_bench_size[1]) * 0.05
                self.max_bench_size[1] = p1_bench_after
                
            if p0_energy_after > self.max_total_energy[0]:
                if acting_player == 0: curr_reward += (p0_energy_after - self.max_total_energy[0]) * 0.05
                self.max_total_energy[0] = p0_energy_after
                
            if p1_energy_after > self.max_total_energy[1]:
                if acting_player == 1: curr_reward += (p1_energy_after - self.max_total_energy[1]) * 0.05
                self.max_total_energy[1] = p1_energy_after
            
            if acting_player == 0:
                reward += (p0_taken * 0.1) - (p1_taken * 0.1) + curr_reward
            else:
                reward += (p1_taken * 0.1) - (p0_taken * 0.1) + curr_reward
                
        return next_state, reward, self.is_done, False, {}

    def close(self):
        battle_finish()
