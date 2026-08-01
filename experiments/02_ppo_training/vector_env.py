import multiprocessing as mp
import os
import sys

def worker(remote, parent_remote, opponent_func):
    parent_remote.close()
    
    # Must append path to find local modules inside worker
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_rule_based')))
    from env_wrapper import PokemonEnvWrapper
    
    env = PokemonEnvWrapper(opponent_func)
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    # Auto-reset for vector environments
                    info['terminal_observation'] = obs
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == 'reset':
                opponent_name = data
                if opponent_name is not None:
                    # Dynamically import the opponent function
                    import importlib
                    module = importlib.import_module(opponent_name)
                    opp_func = getattr(module, opponent_name)
                    obs = env.reset(opponent_func_override=opp_func)
                else:
                    obs = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
    except EOFError:
        pass
    except Exception as e:
        print(f"[Worker Error] {e}")
        remote.send(("error", e))

class VectorEnv:
    def __init__(self, opponent_func, num_envs=16):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [
            mp.Process(target=worker, args=(work_remote, remote, opponent_func))
            for work_remote, remote in zip(self.work_remotes, self.remotes)
        ]
        for p in self.processes:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = []
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
                raise res[1]
            results.append(res)
            
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)

    def reset(self, opponent_name=None):
        for remote in self.remotes:
            remote.send(('reset', opponent_name))
        
        results = []
        for remote in self.remotes:
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "error":
                raise res[1]
            results.append(res)
        return results

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
