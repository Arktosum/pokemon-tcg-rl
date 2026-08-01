"""Diagnostic script: dump observation structure from cabt engine."""
import os
import sys
import json

_this_dir = os.path.dirname(os.path.abspath(__file__))
_agent_dir = os.path.abspath(os.path.join(_this_dir, "..", "01_baseline", "agent"))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from kaggle_environments import make
from cg.api import to_observation_class, OptionType

env = make("cabt", debug=True)
env.reset([os.path.join(_this_dir, "..", "01_baseline", "agent", "main.py"),
           os.path.join(_this_dir, "greedy_bot.py")])

# Get the first observation
obs = env.observation

# Print the raw observation structure
print("=== Raw observation type ===")
print(type(obs))

# Deeply inspect the structure
def inspect(obj, name="", depth=0, max_depth=4):
    if depth > max_depth:
        return
    prefix = "  " * depth
    if isinstance(obj, dict):
        print(f"{prefix}{name}: dict({len(obj)}) keys={list(obj.keys())[:10]}")
        for k, v in list(obj.items())[:3]:
            inspect(v, k, depth+1)
    elif isinstance(obj, list):
        print(f"{prefix}{name}: list({len(obj)})")
        if len(obj) > 0:
            inspect(obj[0], "[0]", depth+1)
    else:
        print(f"{prefix}{name}: {type(obj).__name__} = {repr(obj)[:100]}")

print("\n=== Observation structure ===")
inspect(obs, "obs")

# Try to_observation_class and inspect
print("\n=== After to_observation_class ===")
try:
    obs_obj = to_observation_class(obs)
    print(f"Observation type: {type(obs_obj)}")
    print(f"select type: {type(obs_obj.select)}")
    if obs_obj.select:
        opts = obs_obj.select.option
        print(f"option type: {type(opts)}")
        if isinstance(opts, list) and len(opts) > 0:
            first = opts[0]
            print(f"first option type: {type(first)}")
            print(f"first option has attackId: {hasattr(first, 'attackId')}")
            print(f"first option __dict__: {vars(first) if hasattr(first, '__dict__') else 'N/A'}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
