import os
from kaggle.api.kaggle_api_extended import KaggleApi

def scrape_notebooks():
    api = KaggleApi()
    api.authenticate()
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "community_notebooks")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Fetching top notebooks for pokemon-tcg-ai-battle...")
    notebooks = api.kernels_list(competition='pokemon-tcg-ai-battle', sort_by='voteCount', page_size=20)
    
    print("\n--- Top 3 Downloaded Notebooks ---")
    for i, nb in enumerate(notebooks[:3]):
        print(f"{i+1}. {nb.ref}")
        
    for nb in notebooks[:3]: # Just download the top 3 to save time/bandwidth
        try:
            api.kernels_pull(nb.ref, path=out_dir)
        except Exception as e:
            print(f"Failed to pull {nb.ref}: {e}")
            
    print(f"Notebooks downloaded to {out_dir}")

if __name__ == "__main__":
    scrape_notebooks()
