import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data", "sample_submission", "sample_submission"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from cg.api import to_observation_class
from env import PTCGEnv

with open(os.path.join(os.path.dirname(__file__), "..", "data", "replays", "episode-88673745-replay.json"), 'r') as f:
    d = json.load(f)

printed = 0
for step in d['steps']:
    for agent_step in step:
        if agent_step and 'observation' in agent_step:
            if agent_step['observation'].get('current') is not None:
                if agent_step.get('action') is not None:
                    print("raw_action:", agent_step['action'])
                    printed += 1
                    if printed >= 5:
                        sys.exit(0)
