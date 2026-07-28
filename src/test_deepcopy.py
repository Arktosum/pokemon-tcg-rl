import copy
from env import PTCGEnv

def test():
    env = PTCGEnv()
    obs, _ = env.reset()
    try:
        env2 = copy.deepcopy(env)
        print("DEEPCOPY WORKS")
    except Exception as e:
        print(f"DEEPCOPY FAILED: {e}")

if __name__ == "__main__":
    test()
