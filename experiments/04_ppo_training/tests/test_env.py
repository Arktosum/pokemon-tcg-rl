import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ppo_env import PokemonPPOEnv
from ppo_types import StepResult, ParsedObs

def test_env_reset_and_step():
    env = PokemonPPOEnv(opponent="random", max_steps=5)
    
    reset_res = env.reset()
    assert isinstance(reset_res, StepResult)
    assert reset_res.obs is None or isinstance(reset_res.obs, ParsedObs)
    assert not reset_res.done
    
    # Step 1
    step_res = env.step([0])
    assert isinstance(step_res, StepResult)
    assert isinstance(step_res.done, bool)
    assert isinstance(step_res.reward, float)
