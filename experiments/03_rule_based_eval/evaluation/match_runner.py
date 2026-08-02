import os

def run_single_match(match_id: int, agent1_path: str, agent2_path: str) -> dict:
    devnull_fd = None
    old_stdout = None
    old_stderr = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)

        # Suppress BOTH stdout and stderr at the C-level to silence
        # the OpenSpiel "Unknown game" spam which is emitted via C extensions.
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)

        from kaggle_environments import make
        env = make("cabt")
        steps = env.run([agent1_path, agent2_path])

        turns = len(steps)
        final_step = steps[-1]
        r0 = final_step[0].reward
        r1 = final_step[1].reward

        winner_index = -1
        if r0 is not None and r1 is not None:
            if r0 > r1: winner_index = 0
            elif r1 > r0: winner_index = 1
        elif r0 is not None: winner_index = 0
        elif r1 is not None: winner_index = 1

        return {"match_id": match_id, "winner_index": winner_index, "turns": turns, "error": None}
    except Exception as e:
        return {"match_id": match_id, "winner_index": -1, "turns": 0, "error": str(e)}
    finally:
        # Always restore both file descriptors
        if old_stdout is not None:
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
        if old_stderr is not None:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        if devnull_fd is not None:
            os.close(devnull_fd)
