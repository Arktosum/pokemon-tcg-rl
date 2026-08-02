import multiprocessing as mp
import cloudpickle
from typing import Any, Callable, Dict, List, Tuple
from multiprocessing.connection import Connection
from multiprocessing import Process

class CloudpickleWrapper:
    def __init__(self, var: Any) -> None:
        self.var: Any = var

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs: bytes) -> None:
        self.var = cloudpickle.loads(obs)

def worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper
) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    if info is None:
                        info = {}
                    info['terminal_observation'] = obs
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                if hasattr(env, 'close'):
                    env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command {cmd} not implemented")
    except EOFError:
        pass

class SubprocVecEnv:
    def __init__(self, env_fns: List[Callable[[], Any]]) -> None:
        self.waiting: bool = False
        self.closed: bool = False
        self.num_envs: int = len(env_fns)
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes: List[Process] = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

    def step_async(self, actions: List[Any]) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self) -> Tuple[List[Any], List[float], List[bool], List[Dict[str, Any]]]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)

    def step(self, actions: List[Any]) -> Tuple[List[Any], List[float], List[bool], List[Dict[str, Any]]]:
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> List[Any]:
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return list(results)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
        self.closed = True

    def __del__(self) -> None:
        if not self.closed:
            self.close()
