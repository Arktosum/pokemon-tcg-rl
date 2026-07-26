import sys
import os
import logging

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)

from cg.game import battle_start, battle_select

logger = logging.getLogger(__name__)

class PokemonTCGEnv:
    def __init__(self, deck):
        self.deck = deck
        self.obs_dict = None
        self.state = None
        
    def reset(self):
        """Starts a new game with the provided deck."""
        self.obs_dict, self.state = battle_start(self.deck, self.deck)
        return self.obs_dict
        
    def step(self, action: list[int]):
        """Executes an action and returns (obs, reward, done, info).
        
        info dict contains:
            'termination': 'win' | 'loss' | 'draw' | 'crash' | 'ongoing'
        """
        done = False
        reward = 0.0
        info = {'termination': 'ongoing'}
        
        try:
            self.obs_dict = battle_select(action)
            
            # Check if game ended natively
            result = self.obs_dict['current']['result']
            if result != -1:
                done = True
                # result 0 means player 0 won, 1 means player 1 won, 2 is draw
                if result == 0:
                    reward = 1.0
                    info['termination'] = 'win'
                elif result == 1:
                    reward = -1.0
                    info['termination'] = 'loss'
                else:
                    reward = 0.0
                    info['termination'] = 'draw'
                    
        except Exception as e:
            # Game ended abruptly (IndexError or ValueError from cg.dll)
            # THIS IS A CRASH, NOT A DRAW!
            logger.warning(f"[ENGINE CRASH] cg.dll threw: {type(e).__name__}: {e}")
            self.obs_dict = None
            done = True
            reward = 0.0
            info['termination'] = 'crash'
            
        return self.obs_dict, reward, done
