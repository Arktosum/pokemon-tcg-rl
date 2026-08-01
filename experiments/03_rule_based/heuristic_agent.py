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

def heuristic_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()
        
    ctx = obs.select.context
    options = obs.select.option
    max_count = obs.select.maxCount
    min_count = obs.select.minCount
    
    if max_count == 0 or not options:
        return []

    # Priority 1: Yes/No questions (Always Yes if possible, especially for taking prizes)
    yes_no_contexts = [
        SelectContext.IS_FIRST, SelectContext.MULLIGAN, SelectContext.ACTIVATE, 
        SelectContext.COIN_HEAD, SelectContext.FIRST_EFFECT, SelectContext.MORE_DEVOLVE
    ]
    if ctx in yes_no_contexts or ctx == SelectContext.MAIN:
        for i, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [i]

    if ctx == SelectContext.MAIN:
        # Heuristic scoring
        best_score = -9999
        best_idx = 0
        
        state = obs.current
        my_idx = state.yourIndex if state else 0
        my_state = state.players[my_idx] if state and state.players else None
        
        for i, opt in enumerate(options):
            score = 0
            if opt.type == OptionType.ATTACK:
                score += 1000
            elif opt.type == OptionType.EVOLVE:
                score += 500
            elif opt.type == OptionType.PLAY:
                score += 400
            elif opt.type == OptionType.ATTACH:
                score += 300
            elif opt.type == OptionType.RETREAT:
                if my_state and my_state.active and my_state.active[0]:
                    hp = my_state.active[0].hp
                    max_hp = my_state.active[0].maxHp
                    if max_hp > 0 and hp / max_hp < 0.4 and len(my_state.bench) > 0:
                        score += 800 # High priority retreat
                    else:
                        score -= 500 # Don't retreat randomly
            elif opt.type == OptionType.ABILITY:
                score += 200
            elif opt.type == OptionType.DISCARD:
                score += 10
            elif opt.type == OptionType.END:
                score -= 1000 # Avoid passing turn if we can do something else
                
            if score > best_score:
                best_score = score
                best_idx = i
                
        return [best_idx]
        
    # Default fallback
    try:
        count = max_count if len(options) >= max_count else len(options)
        count = max(count, min_count)
        count = min(count, len(options))
        return random.sample(list(range(len(options))), count)
    except Exception:
        return list(range(min(max_count, len(options))))
