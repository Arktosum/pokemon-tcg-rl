import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent')))
from cg.api import to_observation_class

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

def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()
        
    options = obs.select.option
    max_count = obs.select.maxCount
    if max_count == 0 or not options:
        return []
        
    try:
        count = max_count if len(options) >= max_count else len(options)
        min_count = obs.select.minCount
        count = max(count, min_count)
        count = min(count, len(options))
        return random.sample(list(range(len(options))), count)
    except Exception:
        return list(range(min(max_count, len(options))))
