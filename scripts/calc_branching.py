import json
import glob
files = glob.glob('data/replays/*.json')[:50]
opts = []
for f in files:
    with open(f) as fp:
        for line in fp:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                for s in data:
                    if 'observation' in s and s['observation'].get('current'):
                        sel = s['observation']['current'].get('select')
                        if sel and sel.get('option'):
                            opts.append(len(sel['option']))
            except: pass
print('Total steps:', len(opts))
print('Avg Branching:', sum(opts)/len(opts) if opts else 0)
