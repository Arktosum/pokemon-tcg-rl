import json
from datetime import datetime
import sys
import os

# Suppress noisy OpenSpiel exception print from kaggle_environments import at C-level
fd_out = sys.stdout.fileno()
fd_err = sys.stderr.fileno()
saved_out = os.dup(fd_out)
saved_err = os.dup(fd_err)

devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, fd_out)
os.dup2(devnull, fd_err)

try:
    from kaggle_environments import make, environments
finally:
    os.dup2(saved_out, fd_out)
    os.dup2(saved_err, fd_err)
    os.close(devnull)
    os.close(saved_out)
    os.close(saved_err)

if __name__ == "__main__":
    # print("Available environments:")
    # for env_name in environments:
    #     print(f" - {env_name}")
    
    target_env = "cabt"
    
    import os
    print(f"\nAttempting to make environment: {target_env}")
    try:
        env = make(target_env, debug=True)
        print("Starting battle...")
        agent1 = os.path.join(os.path.dirname(__file__), "agent", "main.py")
        agent2 = os.path.join(os.path.dirname(__file__), "agent", "main.py")
        steps = env.run([agent1, agent2])
        print(f"Battle finished in {len(steps)} steps.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_filename = f"replay_{timestamp}.json"
        with open(replay_filename, "w") as f:
            f.write(json.dumps(env.toJSON()))

        
        print(f"Replay saved to {replay_filename}")
    except Exception as e:
        print(f"Failed to run environment: {e}")
