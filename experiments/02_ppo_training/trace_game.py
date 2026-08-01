"""
Deep trace: print EVERY step's SelectContext, legal count, and prize state.
This will reveal why games end so quickly.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))

from kaggle_environments import make
from agent.cg.api import to_observation_class, SelectContext
from random_agent import random_agent
from aggro_agent import aggro_agent

def _read_deck():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline', 'agent', 'deck.csv'))
    try:
        with open(file_path, "r") as f:
            csv = f.read().split("\n")
        return [int(csv[i]) for i in range(60) if csv[i].strip()]
    except:
        return [1] * 60

def trace_game(opponent_func, opponent_name, max_steps=200):
    env = make("cabt", debug=False)
    deck = _read_deck()
    trainer = env.train([None, opponent_func])
    raw_obs = trainer.reset()
    
    print(f"\n{'='*70}")
    print(f"TRACING GAME vs {opponent_name}")
    print(f"{'='*70}")
    
    step = 0
    
    # Handle initial deck submission
    obs_obj = to_observation_class(raw_obs)
    if obs_obj.select is None:
        print(f"[Step {step:3d}] DECK SUBMISSION (select=None) -> submitting deck")
        raw_obs, r, done, info = trainer.step(deck)
        step += 1
        if done:
            print(f"  DONE after deck submission! reward={r}")
            return
    
    for _ in range(max_steps):
        obs_obj = to_observation_class(raw_obs)
        
        if obs_obj.select is None:
            print(f"[Step {step:3d}] WARNING: obs.select=None mid-game!")
            break
        
        ctx = obs_obj.select.context
        options = obs_obj.select.option
        max_c = obs_obj.select.maxCount
        min_c = obs_obj.select.minCount
        
        # Get prize state
        prize_str = "N/A"
        if obs_obj.current and obs_obj.current.players:
            my_idx = obs_obj.current.yourIndex
            opp_idx = 1 - my_idx
            my_prizes = sum(1 for p in obs_obj.current.players[my_idx].prize if p is not None)
            opp_prizes = sum(1 for p in obs_obj.current.players[opp_idx].prize if p is not None)
            active_hp = "?"
            if obs_obj.current.players[my_idx].active:
                a = obs_obj.current.players[my_idx].active[0]
                if a: active_hp = f"{a.hp}/{a.maxHp}"
            prize_str = f"my_prizes={my_prizes} opp_prizes={opp_prizes} active_hp={active_hp}"
        
        opt_types = [o.type.name if hasattr(o.type, 'name') else str(o.type) for o in options[:5]]
        ctx_str = ctx.name if hasattr(ctx, 'name') else f"INT({ctx})"
        print(f"[Step {step:3d}] ctx={ctx_str:20s} "
              f"n_opts={len(options):3d} min={min_c} max={max_c} "
              f"opts={opt_types} | {prize_str}")
        
        # Pick a safe action (middle option)
        n = len(options)
        count = max(min_c, 1)
        count = min(count, n)
        action = list(range(count))  # Always first N distinct indices
        
        raw_obs, r, done, info = trainer.step(action)
        step += 1
        
        if done:
            # Check final state
            status = env.state[0].status if env.state else "UNKNOWN"
            final_reward = r
            print(f"\n>>> GAME OVER at step {step} | reward={final_reward} status={status}")
            return
    
    print(f"\n>>> Max steps ({max_steps}) reached — game did not finish!")

if __name__ == "__main__":
    trace_game(random_agent, "random_agent", max_steps=200)
    trace_game(aggro_agent, "aggro_agent", max_steps=200)
