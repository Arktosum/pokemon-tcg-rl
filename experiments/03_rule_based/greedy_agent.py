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
        # Dummy deck for testing if file missing
        return [1] * 60

def greedy_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()
        
    ctx = obs.select.context
    options = obs.select.option
    max_count = obs.select.maxCount
    min_count = obs.select.minCount
    
    if max_count == 0:
        return []
        
    if not options:
        return []

    # Priority 1: Yes/No questions (Always be aggressive / say Yes)
    yes_no_contexts = [
        SelectContext.IS_FIRST, SelectContext.MULLIGAN, SelectContext.ACTIVATE, 
        SelectContext.COIN_HEAD, SelectContext.FIRST_EFFECT, SelectContext.MORE_DEVOLVE
    ]
    if ctx in yes_no_contexts:
        for i, opt in enumerate(options):
            if opt.type == OptionType.YES:
                return [i]
        return [0]

    # Priority 2: Main Turn Options
    if ctx == SelectContext.MAIN:
        # We want to do the most impactful thing.
        # EVOLVE > ATTACH > PLAY > ABILITY > ATTACK > END
        priority = {
            OptionType.EVOLVE: 100,
            OptionType.ATTACH: 90,
            OptionType.PLAY: 80,
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
            # If attacking, that's almost always a good greedy terminal action, so bump it slightly 
            # if we can attack, maybe we should? No, developing board is better before attack.
            if score > best_score:
                best_score = score
                best_idx = i
                
        return [best_idx]
        
    # Priority 3: Attach energy
    if ctx in [SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO]:
        # For greedy, just pick the first option (often the active pokemon)
        pass

    # Priority 4: Discarding
    if ctx in [SelectContext.DISCARD, SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD]:
        # Pick randomly if forced to discard
        pass

    # Default fallback: pick random valid indices
    try:
        # If we need to pick EXACTLY max_count
        count = max_count
        if len(options) < max_count:
            count = len(options)
        count = max(count, min_count)
        count = min(count, len(options))
        
        # If we can pick fewer, maybe we want to pick max for greed?
        return random.sample(list(range(len(options))), count)
    except Exception:
        # Absolute safety net
        return list(range(min(max_count, len(options))))

if __name__ == "__main__":
    # Smoke test
    print("Greedy agent loaded successfully.")
