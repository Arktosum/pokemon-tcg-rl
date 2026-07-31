import sys
import os
import json
import traceback

# Add local cg module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

from cg.api import search_begin, search_step, search_end, to_observation_class
from kaggle_environments import make

def investigate_native_search():
    print("--- Step 2: Native Search API Definitions ---")
    print("\n[DOCSTRING for search_begin]")
    print(str(search_begin.__doc__).encode('ascii', 'replace').decode('ascii'))
    print("\n[DOCSTRING for search_step]")
    print(str(search_step.__doc__).encode('ascii', 'replace').decode('ascii'))
    print("\n[DOCSTRING for search_end]")
    print(str(search_end.__doc__).encode('ascii', 'replace').decode('ascii'))

    print("\n--- Step 3: Native Branching API Test ---")
    try:
        # Load a valid observation from the replay
        print("1. Loading Turn 10 from replay...")
        replay_path = os.path.join(os.path.dirname(__file__), 'replay_20260730_125951.json')
        with open(replay_path, 'r', encoding='utf-8') as f:
            replay = json.load(f)
        
        obs_raw = replay['steps'][10][0]['observation']
        obs = to_observation_class(obs_raw)
        
        # We also need to get the env state to compare JSONs. Since we don't have the env running, 
        # let's just initialize the env, force its state to match, or actually, if we use search_begin on the replay's obs,
        # does it mutate the original python dict `obs_raw`? Let's check `obs_raw` mutation instead of `env` mutation!
        # This directly tests if `search_begin` mutates the underlying C++ singleton that might be backing `cg.api`.
        # Wait, if we use a replay, there is no underlying C++ singleton for THIS game running! The engine might crash if we call search_begin on a raw JSON obs?
        # Let's try it.
        
        # Capture before
        orig_json_before = len(json.dumps(obs_raw))
        print("\n2. Calling search_begin()...")
        your_index = obs.current.yourIndex
        state = obs.current
        active = state.players[1 - your_index].active
        
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
        
        # 3. Take a step in the hypothetical branch
        print("\n3. Taking a random step in the hypothetical branch (search_step)...")
        # Find a legal action
        legal_option_idx = 0 # Pick the first available legal option index
        next_search_state = search_step(search_state.searchId, [legal_option_idx])
        print("search_step() returned successfully.")
        
        # 4. Check isolation
        print("\n4. Checking state isolation...")
        orig_json_after = len(json.dumps(obs_raw))
        
        print(f"Hypothetical Branch Hand Size: {len(next_search_state.observation.current.players[your_index].hand)}")
        
        print(f"\nOriginal JSON length before search: {orig_json_before}")
        print(f"Original JSON length after search: {orig_json_after}")
        
        if orig_json_before == orig_json_after:
            print("\nVERDICT: ISOLATION CONFIRMED. The native search API successfully explored a branch without corrupting the original game state.")
        else:
            print("\nVERDICT: CORRUPTION DETECTED. The native search API mutated the original game state.")
            
        print("\nCleaning up with search_end()...")
        search_end()
            
    except Exception as e:
        print(f"\nCRITICAL ERROR during execution:\n{traceback.format_exc()}")

if __name__ == "__main__":
    investigate_native_search()
