import concurrent.futures
import os
import sys

# Adjust sys.path so we can import match_runner even if called from elsewhere
sys.path.insert(0, os.path.dirname(__file__))
from match_runner import run_single_match

def run_tournament(agent1_path: str, agent2_path: str, num_games: int = 10, num_workers: int = 4) -> dict:
    summary = {
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "avg_turns": 0.0,
        "total_errors": 0
    }
    
    total_turns = 0
    valid_games = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_games):
            futures.append(executor.submit(run_single_match, i, agent1_path, agent2_path))
            
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            
            if result["error"] is not None:
                summary["total_errors"] += 1
            else:
                winner = result["winner_index"]
                if winner == 0:
                    summary["agent1_wins"] += 1
                elif winner == 1:
                    summary["agent2_wins"] += 1
                else:
                    summary["draws"] += 1
                    
                total_turns += result["turns"]
                valid_games += 1
                
    if valid_games > 0:
        summary["avg_turns"] = total_turns / valid_games
        
    return summary
