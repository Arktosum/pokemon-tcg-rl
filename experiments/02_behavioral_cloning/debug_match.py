import sys
import os
from kaggle_environments import make

def main():
    agent1 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'titan_agent.py'))
    agent2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent', 'main.py'))

    env = make("cabt", debug=True)
    env.run([agent1, agent2])
    print("Match finished.")
    for i, state in enumerate(env.steps[-1]):
        print(f"Agent {i} status: {state.status}, reward: {state.reward}")
        if 'logs' in state.observation:
            for log in state.observation['logs']:
                print(f"Agent {i} Final Log: {log}")

if __name__ == "__main__":
    main()
