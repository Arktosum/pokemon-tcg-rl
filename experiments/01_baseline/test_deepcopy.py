import copy
import traceback
from kaggle_environments import make
import json

def test_deepcopy_bug():
    print("--- DEEPCOPY BUG REPRODUCTION (TURN 2+) ---")
    
    try:
        env = make("cabt", debug=True)
        # We need to initialize decks to get past turn 0
        dummy_deck = [1] * 60
        # First step is deck submission
        env.reset()
        env.step([dummy_deck, dummy_deck])
        print("1. Original Environment initialized.")
        
        # Take a few random steps to get into the game
        print("\n2. Playing random moves to reach a mid-game state...")
        for _ in range(5):
            if env.done:
                break
            action = []
            for agent_idx in range(2):
                obs_dict = env.state[agent_idx].observation
                if obs_dict.get('select') is None:
                    action.append([])
                elif len(obs_dict.get('select', {}).get('option', [])) > 0:
                    action.append([0]) # Pick first option
                else:
                    action.append([])
            env.step(action)
            
        print("Reached a valid mid-game state.")
        
        orig_json_before = json.dumps(env.state[0].observation)
        
        # Deepcopy the environment
        print("\n3. Attempting to deepcopy the environment...")
        env_copy = copy.deepcopy(env)
        print("Deepcopy successful.")
        
        # Take a step in the COPY
        print("\n4. Taking a real step in the COPY environment...")
        env_copy.step([[0], [0]])
        print("Step in COPY environment successful.")
        
        # Check the states
        print("\n5. Checking state isolation...")
        orig_json_after = json.dumps(env.state[0].observation)
        
        if orig_json_before == orig_json_after:
             print("\nVERDICT: ISOLATION CONFIRMED. Original state remained untouched.")
        else:
             print("\nVERDICT: CORRUPTION DETECTED. Original state mutated when copy was stepped.")
             
    except Exception as e:
        print(f"\nCRITICAL ERROR during execution:\n{traceback.format_exc()}")

if __name__ == "__main__":
    test_deepcopy_bug()
