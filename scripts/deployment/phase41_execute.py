import subprocess
import time

print("1. Submitting Vanilla Baseline...")
submit_cmd = ["kaggle", "competitions", "submit", "-c", "pokemon-tcg-ai-battle", "-f", "submission_vanilla.tar.gz", "-m", "Phase 41 Vanilla"]
result = subprocess.run(submit_cmd, capture_output=True, text=True)
print("Submit Output:", result.stdout)

print("2. Polling Kaggle API...")
time.sleep(5)
leaderboard_cmd = ["kaggle", "competitions", "submissions", "-c", "pokemon-tcg-ai-battle"]
result2 = subprocess.run(leaderboard_cmd, capture_output=True, text=True)
print(result2.stdout)
