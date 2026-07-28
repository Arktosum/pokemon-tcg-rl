import os
import main
from main import agent, MODEL_LOADED

try:
    print("Testing True ONNX Brain Transplant...")
    print(f"Model loaded locally: {MODEL_LOADED}")
    
    # Simulate step = 0 (select is None)
    obs_0 = {"step": 0, "select": None}
    action_0 = agent(obs_0, {})
    print(f"[Step 0] Action Returned Length: {len(action_0)} | Type: {type(action_0)}")

    # Simulate step > 0 (Select with maxCount = 2)
    obs_1 = {"step": 1, "select": {"option": ["A", "B", "C", "D"], "maxCount": 2}}
    action_1 = agent(obs_1, {})
    print(f"[Step 1 maxCount=2] Action Returned: {action_1} | Type: {type(action_1)}")
    
except Exception as e:
    print(f"Fatal error during local validation: {e}")
