import subprocess
import time
import re

def monitor():
    print("Submitting to Kaggle...")
    result = subprocess.run(["kaggle", "competitions", "submit", "-c", "pokemon-tcg-ai-battle", "-f", "submission.tar.gz", "-m", "ONNX True Deployment Phase 33"], capture_output=True, text=True)
    print(result.stdout)
    
    # We will just parse the list of submissions and wait for our latest one to change from PENDING
    attempts = 0
    while attempts < 10:
        res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "pokemon-tcg-ai-battle"], capture_output=True, text=True)
        lines = res.stdout.split('\n')
        data_lines = [l for l in lines if "ONNX True Deployment" in l]
        if data_lines:
            latest = data_lines[0]
            parts = latest.split()
            sub_id = parts[0]
            if "PENDING" not in latest:
                print(f"Final Status Reached! Submission ID: {sub_id}")
                if "COMPLETE" in latest:
                    print("Status: COMPLETE")
                elif "ERROR" in latest:
                    print("Status: ERROR")
                else:
                    print(f"Status: {latest}")
                break
            else:
                print(f"Submission {sub_id} is PENDING. Waiting 15 seconds...")
        else:
            print("Submission not found yet in list...")
        time.sleep(15)
        attempts += 1

if __name__ == "__main__":
    monitor()
