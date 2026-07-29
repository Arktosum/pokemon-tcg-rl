import os
import sys
import random
import traceback

from cg.api import Observation, to_observation_class

model = None

def read_deck_csv() -> list[int]:
    """Read deck.csv.
    
    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck

def _process_obs(obs):
    import numpy as np
    MAX_OPTIONS = 500
    state_vec = np.zeros(120, dtype=np.float32)
    mask = np.zeros(MAX_OPTIONS, dtype=np.int8)
    
    if obs.select is not None:
        valid_len = min(len(obs.select.option), MAX_OPTIONS)
        mask[:valid_len] = 1
        
    if obs.current is not None:
        state = obs.current
        state_vec[0] = state.turn
        state_vec[1] = state.turnActionCount
        state_vec[2] = state.yourIndex
        state_vec[3] = state.firstPlayer
        
        idx = 4
        for p_idx in [0, 1]:
            p_state = state.players[p_idx]
            state_vec[idx] = p_state.deckCount / 60.0; idx += 1
            state_vec[idx] = p_state.handCount / 60.0; idx += 1
            state_vec[idx] = p_state.benchMax / 5.0; idx += 1
            state_vec[idx] = len(p_state.prize) / 6.0; idx += 1
            state_vec[idx] = float(p_state.poisoned); idx += 1
            state_vec[idx] = float(p_state.burned); idx += 1
            state_vec[idx] = float(p_state.asleep); idx += 1
            state_vec[idx] = float(p_state.paralyzed); idx += 1
            state_vec[idx] = float(p_state.confused); idx += 1
            
            if p_state.active and p_state.active[0]:
                pk = p_state.active[0]
                state_vec[idx] = pk.id; idx += 1
                state_vec[idx] = pk.hp / 350.0; idx += 1
                state_vec[idx] = pk.maxHp / 350.0; idx += 1
                state_vec[idx] = len(pk.energies) / 10.0; idx += 1
            else:
                idx += 4
                
            for b_i in range(5):
                if b_i < len(p_state.bench):
                    pk = p_state.bench[b_i]
                    state_vec[idx] = pk.id; idx += 1
                    state_vec[idx] = pk.hp / 350.0; idx += 1
                    state_vec[idx] = pk.maxHp / 350.0; idx += 1
                    state_vec[idx] = len(pk.energies) / 10.0; idx += 1
                else:
                    idx += 4
                    
    return state_vec, mask

def agent(obs_dict: dict) -> list[int]:
    """Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    global model
    obs: Observation = to_observation_class(obs_dict)
    
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        sys.stderr.write("TITAN TELEMETRY: Agent Initialization / Step 0 Deck Query.\n")
        return read_deck_csv()
    
    try:
        import torch
        import numpy as np
        
        if model is None:
            # Safely add /kaggle_simulations/agent/ to sys.path to import model.py
            sys.path.append("/kaggle_simulations/agent/")
            sys.path.append(os.getcwd())
            
            from model import PokemonActorCritic
            sys.stderr.write("TITAN TELEMETRY: Instantiating PokemonActorCritic(num_layers=3)...\n")
            model = PokemonActorCritic(num_layers=3)
            
            model_path = "/kaggle_simulations/agent/TOP_ELO_BC_MODEL_FINAL.pt"
            if not os.path.exists(model_path):
                model_path = os.path.join(os.getcwd(), "checkpoints", "TOP_ELO_BC_MODEL_FINAL.pt")
                if not os.path.exists(model_path):
                    model_path = os.path.join(os.getcwd(), "TOP_ELO_BC_MODEL_FINAL.pt")
                
            if os.path.exists(model_path):
                sys.stderr.write(f"TITAN TELEMETRY: Loading weights from {model_path}...\n")
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                sys.stderr.write("TITAN TELEMETRY: Weights loaded successfully.\n")
            model.eval()

        state_vec, mask = _process_obs(obs)
        sys.stderr.write(f"TITAN TELEMETRY: Observation parsed. Branching factor: {len(obs.select.option)}\n")
        
        s_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        m_tensor = torch.tensor(mask, dtype=torch.int8).unsqueeze(0)
        
        with torch.no_grad():
            policy, _ = model(s_tensor, m_tensor)
            
        sys.stderr.write("TITAN TELEMETRY: Neural network forward pass completed.\n")
        
        p = policy.squeeze(0).cpu().numpy()
        valid_actions = np.where(np.array(mask) == 1)[0]
        
        num_opts = len(obs.select.option)
        min_c = obs.select.minCount if obs.select.minCount is not None else 1
        max_c = obs.select.maxCount if obs.select.maxCount is not None else 1
        
        # VERY IMPORTANT: If max_c is 0, we MUST return an empty list immediately!
        if max_c == 0 or num_opts == 0:
            return []
        
        if len(valid_actions) > 0:
            p_valid = p[valid_actions]
            if p_valid.sum() > 0:
                p_valid /= p_valid.sum()
                action = int(np.random.choice(valid_actions, p=p_valid))
            else:
                action = int(np.random.choice(valid_actions))
        else:
            # Fallback to random if no valid actions found from mask
            action = random.randint(0, num_opts - 1)
            
        select_list = [action]
        available = list(range(num_opts))
        if action in available:
            available.remove(action)
            
        # Pad up to minCount (not maxCount) to satisfy requirements without making excessive random choices
        while len(select_list) < min_c and available:
            nxt = random.choice(available)
            select_list.append(nxt)
            available.remove(nxt)
            
        return select_list
        
    except Exception as e:
        err_msg = traceback.format_exc()
        sys.stderr.write(f"TITAN TELEMETRY FATAL EXCEPTION:\n{err_msg}\n")
        # Fallback EXACTLY like sample submission
        num_opts = len(obs.select.option)
        max_c = obs.select.maxCount if obs.select.maxCount is not None else 1
        if max_c == 0 or num_opts == 0:
            return []
        
        # Ensure we don't try to sample more than available options
        sample_count = min(num_opts, max_c)
        return random.sample(list(range(num_opts)), sample_count)
