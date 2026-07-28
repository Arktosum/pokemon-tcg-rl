import subprocess
import json
import time
import sys

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception as e:
        return str(e)

print("1. Repackaging and Submitting...")
run_cmd("python scripts/package_onnx_submission.py")
submit_out = run_cmd("kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz -m \"Phase 40 Valid\"")
print(f"Submit Output: {submit_out}")

print("2. Telemetry Reboot...")
leaderboard = run_cmd("kaggle competitions leaderboard pokemon-tcg-ai-battle --show")
print(f"Leaderboard Output: {leaderboard}")

print("3. Randomness Audit...")
# we can list episodes for our team
episodes = run_cmd("kaggle competitions episodes pokemon-tcg-ai-battle")
print(f"Episodes Output: {episodes}")
