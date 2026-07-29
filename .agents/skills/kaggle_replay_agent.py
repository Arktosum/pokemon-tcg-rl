import json
import numpy as np
import os
import glob

class KaggleReplayAgent:
    def __init__(self, replays_dir="."):
        self.replays_dir = replays_dir
        self.replays = []
        self._load_replays()
        
    def _load_replays(self):
        # Scan for Kaggle JSON replay files
        pattern = os.path.join(self.replays_dir, "*-replay.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    self.replays.append(data)
            except Exception as e:
                print(f"Failed to load replay {filepath}: {e}")

    def act(self, obs, env=None):
        mask = obs.get("action_mask", [])
        valid_actions = np.where(np.array(mask) == 1)[0]
        
        if len(valid_actions) == 0:
            return 0
            
        # Ingest logic: normally we would parse obs and match to replay states.
        # Fallback to random uniform selection from valid actions if we can't map exact states.
        # This acts as a foundation for behavioral cloning from replays.
        return int(np.random.choice(valid_actions))
