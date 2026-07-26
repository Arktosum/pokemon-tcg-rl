import zipfile
import json
import os

zip_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\replays\pokemon-tcg-ai-battle-episodes-2026-07-12.zip'

if not os.path.exists(zip_path):
    print("Zip file not found.")
    exit(1)

with zipfile.ZipFile(zip_path, 'r') as z:
    names = z.namelist()
    json_names = [n for n in names if n.endswith('.json')]
    print(f"Found {len(json_names)} JSON files in archive.")
    
    if json_names:
        sample_name = json_names[0]
        with z.open(sample_name) as f:
            data = json.load(f)
            print(f"Keys in JSON: {data.keys()}")
            
            steps = data.get('steps', [])
            print(f"Number of steps: {len(steps)}")
            if len(steps) > 5:
                # Print action format for step 5
                print(f"Step 5 Player 0 action: {steps[5][0].get('action')}")
                print(f"Step 5 Player 1 action: {steps[5][1].get('action')}")
                
                # Check if observation has what we expect
                obs0 = steps[5][0].get('observation', {})
                print(f"Step 5 Player 0 Observation Keys: {obs0.keys()}")
