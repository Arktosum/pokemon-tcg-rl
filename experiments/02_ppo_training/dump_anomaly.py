import os, json

dataset_dir = 'experiments/01_baseline/dataset/matches'
json_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.json')]

for fpath in json_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        replay = json.load(f)
        
    rewards = replay.get('rewards', [])
    if not rewards or max(rewards) <= 0: continue
    winner_index = rewards.index(max(rewards))
    
    for step_idx, step_data in enumerate(replay.get('steps', [])):
        if len(step_data) <= winner_index: continue
        agent_data = step_data[winner_index]
        obs_dict = agent_data.get('observation')
        action = agent_data.get('action')
        
        if not obs_dict or action is None: continue
        if 'select' not in obs_dict or obs_dict['select'] is None: continue
        if len(action) == 60: continue # SKIP DECK SUBMISSIONS
        
        select = obs_dict['select']
        legal_option_count = len(select.get('option', []))
        
        t = action[0] if (isinstance(action, list) and len(action) > 0) else 0
        if t >= legal_option_count:
            print(f'Anomaly found in {fpath} at step {step_idx}')
            print(f'action: {action}')
            print(f'legal_option_count: {legal_option_count}')
            print(f'Context: {select.get("context")}')
            print(f'Options: {json.dumps(select.get("option", []))}')
            exit(0)
