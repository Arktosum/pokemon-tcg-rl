import json
with open('experiments/01_baseline/dataset/matches/88887869.json', 'r', encoding='utf-8') as f:
    replay = json.load(f)
for i in range(2):
    print(f'--- STEP {i} ---')
    obs = replay['steps'][i][0]['observation']
    act = replay['steps'][i][0]['action']
    select = obs.get('select')
    print(f'obs.select.context: {select.get("context") if select else None}')
    print(f'obs.select.option length: {len(select.get("option", [])) if select else 0}')
    print(f'action length: {len(act) if act else 0}')
    print(f'action: {act}')
