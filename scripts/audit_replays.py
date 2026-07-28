import os
import json
import csv

def audit():
    replay_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "replays")
    out_csv = os.path.join(os.path.dirname(__file__), "replay_metadata.csv")
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Winner_Team_Name", "Winner_Elo", "Number_of_Steps"])
        
        for filename in os.listdir(replay_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(replay_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
                
            info = data.get('info', {})
            rewards = data.get('rewards', [0, 0])
            team_names = info.get('TeamNames', ['Unknown', 'Unknown'])
            
            winner_idx = 0 if rewards[0] > rewards[1] else 1
            winner_team = team_names[winner_idx]
            
            winner_elo = "N/A"
            
            steps = data.get('steps', [])
            num_steps = len(steps)
            
            writer.writerow([filename, winner_team, winner_elo, num_steps])
            safe_team = winner_team.encode('ascii', 'replace').decode('ascii')
            print(f"Processed {filename}: Winner={safe_team}, Steps={num_steps}")

if __name__ == "__main__":
    audit()
