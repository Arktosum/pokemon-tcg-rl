import multiprocessing as mp
from typing import Callable, Any, List
from vec_env import SubprocVecEnv
from env_wrapper import PokemonPPOEnv

def make_env() -> Callable[[], Any]:
    def _thunk() -> Any:
        return PokemonPPOEnv()
    return _thunk

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    env_fns: List[Callable[[], Any]] = [make_env() for _ in range(4)]
    env: SubprocVecEnv = SubprocVecEnv(env_fns)
    
    print("Resetting environment...")
    obs = env.reset()
    assert len(obs) == 4, f"Expected 4 observations, got {len(obs)}"
    print("Env reset successfully, got 4 observations.")
    
    for i in range(20):
        actions: List[List[int]] = [[0], [0], [0], [0]]
        obs, rewards, dones, infos = env.step(actions)
        print(f"Step {i+1}: Dones={dones}, Rewards={rewards}")
        
    env.close()
    print("Env closed successfully.")
