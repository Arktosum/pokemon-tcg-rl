import zipfile
import json
import os

zip_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\replays\pokemon-tcg-ai-battle-episodes-2026-07-12.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    json_names = [n for n in z.namelist() if n.endswith('.json')]
    with z.open(json_names[0]) as f:
        data = json.load(f)
        print("Replay ID:", data.get('id'))
        print("Configuration keys:", data.get('configuration', {}).keys())
        print("Info keys:", data.get('info', {}).keys())
        
        # Look for deck info in info or configuration
        if 'info' in data:
            print("Info:", data['info'])
            
        # Maybe step 1 or 2 has the deck in `current.players.deck`?
        for s in data['steps'][:10]:
            obs = s[0].get('observation', {})
            if obs and obs.get('current'):
                p0 = obs['current']['players'][0]
                # print the type of p0['deck']
                print("Deck type:", type(p0.get('deck')))
                if isinstance(p0.get('deck'), list) and len(p0.get('deck')) > 0:
                    print("Deck cards:", p0.get('deck')[:5], "...")
                    break
