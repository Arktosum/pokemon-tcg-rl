import os
import sys

def main():
    agent1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "agents/dragapult.py"))
    agent2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../01_baseline/agent/main.py"))
    
    from kaggle_environments import make
    env = make("cabt", debug=True)
    steps = env.run([agent1_path, agent2_path])
    
    print("Match finished with", len(steps), "steps.")
    final_step = steps[-1]
    
    obs = final_step[0].observation
    # The observation is a dict, it might contain "logs" or we might need to parse it
    # We can just print the observation
    import json
    # Print the last few steps
    for step_idx in range(max(0, len(steps)-3), len(steps)):
        print(f"--- STEP {step_idx} ---")
        st = steps[step_idx]
        for p in st:
            obs_dict = p.observation
            if "logs" in obs_dict:
                for log in obs_dict["logs"]:
                    if log.get("type") == 23:
                        print("RESULT LOG:", log)
                        
if __name__ == "__main__":
    main()
