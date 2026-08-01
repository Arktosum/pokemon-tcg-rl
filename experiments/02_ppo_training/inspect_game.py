"""
Deep dive: inspect raw prize objects, context int->name mapping, and option types.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))

from kaggle_environments import make
from agent.cg.api import to_observation_class, SelectContext, OptionType
from random_agent import random_agent

def _read_deck():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent', 'deck.csv'))
    try:
        with open(file_path, "r") as f:
            lines = f.read().split("\n")
        return [int(lines[i]) for i in range(60) if lines[i].strip()]
    except:
        return [1] * 60

print("=== SelectContext enum values ===")
for name, val in SelectContext.__members__.items():
    print(f"  {name} = {val.value}")

print("\n=== OptionType enum values ===")
for name, val in OptionType.__members__.items():
    print(f"  {name} = {val.value}")

print("\n=== Running game trace ===")
env = make("cabt", debug=False)
deck = _read_deck()
trainer = env.train([None, random_agent])
raw_obs = trainer.reset()

obs_obj = to_observation_class(raw_obs)
if obs_obj.select is None:
    print("[Step 0] DECK SUBMISSION -> submitting deck")
    raw_obs, r, done, info = trainer.step(deck)

for step in range(1, 30):
    obs_obj = to_observation_class(raw_obs)
    if obs_obj.select is None:
        print(f"[Step {step}] select=None! Breaking.")
        break
    
    ctx = obs_obj.select.context
    opts = obs_obj.select.option
    
    # Inspect prize zone
    prize_raw = []
    if obs_obj.current and obs_obj.current.players:
        my_idx = obs_obj.current.yourIndex
        opp_idx = 1 - my_idx
        my_p = obs_obj.current.players[my_idx]
        opp_p = obs_obj.current.players[opp_idx]
        # Print actual prize list
        my_prize_list = [str(p) if p is not None else "None" for p in (my_p.prize or [])]
        opp_prize_list = [str(p) if p is not None else "None" for p in (opp_p.prize or [])]
        prize_raw = f"my_prize_list(len={len(my_p.prize or [])}): {my_prize_list[:3]}... | opp(len={len(opp_p.prize or [])}): {opp_prize_list[:3]}..."
        
        # Also check bench
        bench_count = len(my_p.bench) if my_p.bench else 0
        active_str = f"{my_p.active[0].hp}/{my_p.active[0].maxHp}" if my_p.active and my_p.active[0] else "?"
        hand_count = len(my_p.hand) if my_p.hand else 0
    
    opt_types_raw = [(o.type, getattr(o.type, 'value', o.type)) for o in opts[:3]]
    ctx_name = ctx.name if hasattr(ctx, 'name') else f"INT({ctx})"
    ctx_val = ctx.value if hasattr(ctx, 'value') else int(ctx)
    
    print(f"\n[Step {step}] ctx={ctx_name}(val={ctx_val}) | {len(opts)} opts | min={obs_obj.select.minCount} max={obs_obj.select.maxCount}")
    print(f"  opt_types (raw): {opt_types_raw}")
    print(f"  active={active_str} bench={bench_count} hand={hand_count}")
    print(f"  prizes: {prize_raw}")
    
    # Always pick first valid action
    count = max(obs_obj.select.minCount, 1)
    count = min(count, len(opts))
    raw_obs, r, done, info = trainer.step(list(range(count)))
    
    if done:
        status = env.state[0].status if env.state else "?"
        print(f"\n>>> GAME OVER at step {step+1} | reward={r} | status={status}")
        # Print final state
        obs_obj2 = to_observation_class(raw_obs) if raw_obs else None
        break
