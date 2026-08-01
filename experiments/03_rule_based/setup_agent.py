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

def setup_agent(obs_dict: dict) -> list[int]:
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

    if ctx == SelectContext.MAIN:
        # Setup priority: Play cards (draw/setup) and evolve before attacking.
        # PLAY > EVOLVE > ATTACH > ABILITY > ATTACK > RETREAT
        priority = {
            OptionType.PLAY: 100,
            OptionType.EVOLVE: 90,
            OptionType.ATTACH: 80,
            OptionType.ABILITY: 70,
            OptionType.ATTACK: 60,
            OptionType.RETREAT: 10,
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
        
    try:
        count = max_count if len(options) >= max_count else len(options)
        count = max(count, min_count)
        count = min(count, len(options))
        return random.sample(list(range(len(options))), count)
    except Exception:
        return list(range(min(max_count, len(options))))
