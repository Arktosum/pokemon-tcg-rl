import json

with open('episode-88792394-replay.json') as f:
    data = json.load(f)

# Confirm: action[0] is the INDEX into select.option array
step15 = data['steps'][15]
ag0 = step15[0]
obs = ag0['observation']
sel = obs['select']
opts = sel.get('option', [])
print('Step 15 Ag0: action=', ag0['action'])
print('  option[6]:', opts[6] if len(opts) > 6 else 'N/A')
print('  total opts:', len(opts))

print('\nAgent 1 active steps with action:')
for i, step in enumerate(data['steps']):
    if not isinstance(step, list) or len(step) < 2:
        continue
    ag = step[1]
    if not ag:
        continue
    status = ag.get('status')
    action = ag.get('action')
    obs = ag.get('observation', {})
    sel = obs.get('select')
    if status == 'ACTIVE' and action and len(action) > 0:
        n_opts = len(sel.get('option', [])) if isinstance(sel, dict) else 0
        print(f'  Step {i}: action={action}, n_opts={n_opts}')
