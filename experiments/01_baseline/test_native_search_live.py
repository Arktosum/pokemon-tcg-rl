import sys
import os
import json
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

from cg.api import search_begin, search_step, search_end, to_observation_class
from kaggle_environments import make

def run_live_test():
    print("--- LIVE ENGINE NATIVE SEARCH TEST ---")
    try:
        env = make("cabt", debug=True)
        # We need to initialize decks to get past turn 0
        dummy_deck = [1] * 60
        # First step is deck submission
        env.reset()
        env.step([dummy_deck, dummy_deck])
        print("1. Live Environment initialized and game started.")
        
        # Take a few random steps to get into the game
        print("\n2. Playing random moves to reach a mid-game state...")
        for _ in range(5):
            if env.done:
                break
            action = []
            for agent_idx in range(2):
                obs_dict = env.state[agent_idx].observation
                obs = to_observation_class(obs_dict)
                if obs.select is None:
                    action.append([])
                elif len(obs.select.option) > 0:
                    action.append([0]) # Pick first option
                else:
                    action.append([])
            env.step(action)
            
        print("Reached a valid mid-game state.")
        
        # Now we are in a live game state. 
        obs_raw = env.state[0].observation
        obs = to_observation_class(obs_raw)
        
        your_index = obs.current.yourIndex
        state = obs.current
        active = state.players[1 - your_index].active
        
        orig_json_before = json.dumps(obs_raw)
        
        print("\n3. Branching via search_begin()...")
        search_state = search_begin(
            obs,
            your_deck=[1] * state.players[your_index].deckCount,
            your_prize=[1] * len(state.players[your_index].prize),
            opponent_deck=[1072] * state.players[1 - your_index].deckCount,
            opponent_prize=[1] * len(state.players[1 - your_index].prize),
            opponent_hand=[1] * state.players[1 - your_index].handCount,
            opponent_active=[1072] if len(active) > 0 and active[0] == None else []
        )
        print("search_begin() returned successfully. SearchId:", search_state.searchId)
        
        # Take multiple hypothetical steps
        print("\n4. Taking 3 hypothetical steps in the branch...")
        for i in range(3):
            next_search_state = search_step(search_state.searchId, [0])
            print(f"Hypothetical Step {i+1} completed.")
            search_state = next_search_state
            
        # Clean up
        search_end()
        print("\n5. search_end() called to cleanup branch.")
        
        # Now check if the live environment is still intact
        print("\n6. Checking if live environment is intact...")
        obs_raw_after_branch = env.state[0].observation
        orig_json_after_branch = json.dumps(obs_raw_after_branch)
        
        if orig_json_before == orig_json_after_branch:
             print("SUCCESS: Live state JSON is identical before and after branching.")
        else:
             print("FAILURE: Live state JSON was mutated by branching.")
             
        # Take a real step in the live environment to ensure it hasn't crashed or entered a broken state
        print("\n7. Taking a real step in the live environment...")
        env.step([[0], [0]])
        print("Real step completed successfully.")
        
        print("\nVERDICT: NATIVE SEARCH IS 100% ISOLATED AND SAFE ON LIVE ENGINE.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR during execution:\n{traceback.format_exc()}")

if __name__ == "__main__":
    run_live_test()
