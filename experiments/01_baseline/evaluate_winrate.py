import os
import multiprocessing
from kaggle_environments import make

def run_match(args):
    match_id, agent1, agent2 = args
    env = make("cabt", debug=False)
    
    # We pass the paths to the agents
    steps = env.run([agent1, agent2])
    
    # Analyze the result
    final_step = steps[-1]
    
    p1_status = final_step[0].status
    p2_status = final_step[1].status
    
    p1_reward = final_step[0].reward
    p2_reward = final_step[1].reward
    
    # Kaggle environments sets status to 'ERROR' when a timeout or crash occurs
    # It sets status to 'DONE' when the game ends normally.
    
    if p1_status == 'ERROR' or p2_status == 'ERROR':
        # One or both agents crashed or timed out
        if p1_status == 'ERROR' and p2_status == 'ERROR':
            return 'ERROR_BOTH'
        elif p1_status == 'ERROR':
            return 'ERROR_P1'
        else:
            return 'ERROR_P2'
            
    if p1_status == 'INVALID' or p2_status == 'INVALID':
        if p1_status == 'INVALID' and p2_status == 'INVALID':
            return 'INVALID_BOTH'
        elif p1_status == 'INVALID':
            return 'INVALID_P1'
        else:
            return 'INVALID_P2'
    
    if p1_reward == 1 and p2_reward == -1:
        return 'P1_WIN'
    elif p1_reward == -1 and p2_reward == 1:
        return 'P2_WIN'
    else:
        # A true draw based on rewards
        return 'DRAW'

def evaluate_agents(agent1, agent2, num_matches=1000, num_workers=10):
    print(f"Evaluating {agent1} vs {agent2} for {num_matches} matches...")
    
    args = [(i, agent1, agent2) for i in range(num_matches)]
    
    results = {
        'P1_WIN': 0,
        'P2_WIN': 0,
        'DRAW': 0,
        'ERROR_P1': 0,
        'ERROR_P2': 0,
        'ERROR_BOTH': 0,
        'INVALID_P1': 0,
        'INVALID_P2': 0,
        'INVALID_BOTH': 0
    }
    
    with multiprocessing.Pool(num_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_match, args)):
            results[res] += 1
            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{num_matches}] Current Results: {results}")
                
    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agent_mcts = os.path.join(base_dir, "agent_mcts_eval.py")
    agent_random = "random"
    
    print("--- EVALUATION 1: MCTS vs Random ---")
    results_mcts_vs_random = evaluate_agents(agent_mcts, agent_random, num_matches=1000)
    print("Final Results MCTS vs Random:")
    print(results_mcts_vs_random)
    
    print("\n--- EVALUATION 2: MCTS vs MCTS ---")
    results_mcts_vs_mcts = evaluate_agents(agent_mcts, agent_mcts, num_matches=1000)
    print("Final Results MCTS vs MCTS:")
    print(results_mcts_vs_mcts)
