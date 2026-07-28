import os
import sys

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_submission", "sample_submission")
if ENGINE_PATH not in sys.path:
    sys.path.append(ENGINE_PATH)

from cg.game import battle_start, battle_select
from cg.api import to_observation_class

def test_select():
    with open(os.path.join(ENGINE_PATH, "deck.csv"), "r") as f:
        deck = [int(x) for x in f.read().strip().split("\n")[:60]]
        
    obs_dict, _ = battle_start(deck, deck)
    obs = to_observation_class(obs_dict)
    
    for _ in range(20):
        if obs.select is not None:
            print("Select Options:", obs.select.option)
            action = [0]
            if "Attack" in [str(x) for x in obs.select.option]:
                print("FOUND ATTACK!")
        else:
            print("No select options.")
            break
            
        try:
            obs_dict = battle_select([0])
            obs = to_observation_class(obs_dict)
        except:
            break

test_select()
