import os
import csv
import json
import kaggle
import time
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    manifest_path = os.path.join(dataset_dir, "manifest.csv")
    
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Please run 'kaggle datasets download -d kaggle/pokemon-tcg-ai-battle-episodes-index --unzip' first.")
        return
        
    print("Reading manifest.csv to find the most recent high-ELO daily export...")
    latest_slug = None
    latest_date = None
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row['date']
            slug = row['daily_dataset_slug']
            
            # Keep track of the most recent dataset
            if latest_date is None or date > latest_date:
                latest_date = date
                latest_slug = slug
                
    if not latest_slug:
        print("No daily exports found in manifest.")
        return
        
    print(f"Found latest top-tier dataset export: {latest_slug} (Date: {latest_date})")
    
    # We will use a local JSON tracker to avoid re-downloading if we already have it.
    tracker_path = os.path.join(dataset_dir, "downloaded_episodes.json")
    downloaded = {}
    if os.path.exists(tracker_path):
        with open(tracker_path, 'r') as f:
            downloaded = json.load(f)
            
    if downloaded.get(latest_slug):
        print(f"Dataset {latest_slug} is already downloaded and tracked. Skipping download.")
        return
        
    matches_dir = os.path.join(dataset_dir, "matches")
    os.makedirs(matches_dir, exist_ok=True)
    
    print(f"Downloading {latest_slug} to {matches_dir}...")
    
    # Exponential backoff wrapper
    max_retries = 5
    base_delay = 5
    for attempt in range(max_retries):
        try:
            api.dataset_download_files(
                f"kaggle/{latest_slug}",
                path=matches_dir,
                unzip=True,
                quiet=False
            )
            print("Download and extraction complete!")
            # Mark as downloaded
            downloaded[latest_slug] = True
            with open(tracker_path, 'w') as f:
                json.dump(downloaded, f)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Download failed.")
                raise e
            delay = base_delay * (2 ** attempt)
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

if __name__ == "__main__":
    main()
