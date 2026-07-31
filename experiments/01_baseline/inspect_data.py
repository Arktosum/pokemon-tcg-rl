import os
import glob
import random
import json

replay_dir = "experiments/01_baseline/dataset/matches"
files = glob.glob(os.path.join(replay_dir, "*.json"))

print(f"Total files found: {len(files)}")
unique_files = set(files)
print(f"Unique files: {len(unique_files)}")

print("\n--- 5 Random Filenames ---")
sample_files = random.sample(files, min(5, len(files)))
for f in sample_files:
    print(os.path.basename(f))

print("\n--- Duplicate Spot Check (Match IDs) ---")
for f in sample_files:
    with open(f, 'r') as fp:
        try:
            data = json.load(fp)
            match_id = data.get('id', 'N/A')
            print(f"{os.path.basename(f)} -> Match ID: {match_id}")
        except:
            print(f"{os.path.basename(f)} -> Error parsing JSON")
