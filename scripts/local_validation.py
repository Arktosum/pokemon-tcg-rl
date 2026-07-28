import sys
import os

try:
    import main
    from main import agent
    print("Agent imported successfully.")
    
    # Assert global model loaded successfully
    assert hasattr(main, 'MODEL_LOADED_SUCCESSFULLY') and main.MODEL_LOADED_SUCCESSFULLY, "Model did not load successfully in main.py global scope!"
    print("Assertion passed: Transformer model loaded successfully.")
    
    obs = {"step": 0, "remainingOverageTime": 60, "action": []}
    conf = {}
    
    action = agent(obs, conf)
    print("Agent returned action:", action)
except AssertionError as ae:
    print(f"Fatal AssertionError: {ae}")
    sys.exit(1)
except Exception as e:
    print(f"Fatal Error during initialization or execution: {e}")
    sys.exit(1)
