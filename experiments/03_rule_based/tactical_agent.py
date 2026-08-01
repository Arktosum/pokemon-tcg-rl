import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent')))
from cg.api import Observation, to_observation_class, SelectContext, OptionType

def read_deck_csv() -> list[int]:
    file_path = os.path.join(os.path.dirname(__file__), "..", "01_baseline", "agent", "deck.csv")
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

def tactical_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()
        
    ctx = obs.select.context
    options = obs.select.option
    max_count = obs.select.maxCount
    min_count = obs.select.minCount
    
    if max_count == 0 or not options:
        return []

    # Priority 1: Yes/No questions
    yes_no_contexts = [
        SelectContext.IS_FIRST, SelectContext.MULLIGAN, SelectContext.ACTIVATE, 
        SelectContext.COIN_HEAD, SelectContext.FIRST_EFFECT, SelectContext.MORE_DEVOLVE
    ]
    if ctx in yes_no_contexts:
        for i, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [i]
        return [0]

    # Priority 2: Main Turn Options - Tactical
    if ctx == SelectContext.MAIN:
        # Tactical logic: check if Active Pokemon is low HP to force retreat
        needs_retreat = False
        state = obs.current
        if state and state.players:
            my_state = state.players[state.yourIndex]
            if my_state.active and my_state.active[0]:
                active_hp = my_state.active[0].hp
                active_max = my_state.active[0].maxHp
                # Retreat if below 30% HP and bench is available
                if active_max > 0 and (active_hp / active_max) < 0.3 and len(my_state.bench) > 0:
                    needs_retreat = True
                    
        priority = {
            OptionType.EVOLVE: 100,
            OptionType.ATTACH: 90,
            OptionType.PLAY: 80,
            OptionType.ABILITY: 70,
            OptionType.ATTACK: 60,
            OptionType.RETREAT: 150 if needs_retreat else 10,  # Tactical retreat!
            OptionType.DISCARD: 5,
            OptionType.END: 0
        }
        
        best_score = -1
        best_idx = 0
        for i, opt in enumerate(options):
            score = priority.get(opt.type, 0)
            if score > best_score:
                best_score = score
                best_idx = i
                
        return [best_idx]
        
    # Default fallback: pick random valid indices
    try:
        count = max_count if len(options) >= max_count else len(options)
        count = max(count, min_count)
        count = min(count, len(options))
        return random.sample(list(range(len(options))), count)
    except Exception:
        return list(range(min(max_count, len(options))))

if __name__ == "__main__":
    print("Tactical agent loaded successfully.")
