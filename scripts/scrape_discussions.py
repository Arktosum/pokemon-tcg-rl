import os
from kaggle.api.kaggle_api_extended import KaggleApi

def scrape_discussions():
    api = KaggleApi()
    api.authenticate()
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    dump_path = os.path.join(out_dir, "community_discussions_dump.txt")
    
    print("Fetching top discussion topics for pokemon-tcg-ai-battle...")
    try:
        res = api.competition_list_topics('pokemon-tcg-ai-battle')
        topics = res.topics
        with open(dump_path, "w", encoding="utf-8") as f:
            for t in topics:
                f.write(f"Title: {t.title}\n")
                f.write(f"Author: {t.author_name}\n")
                f.write(f"Replies: {t.comment_count}\n")
                f.write(f"URL: {t.topic_url}\n")
                f.write("-" * 40 + "\n")
                
        print(f"Saved topics to {dump_path}")
        
        print("\n--- Top 5 Discussion Topics ---")
        for i, t in enumerate(topics[:5]):
            print(f"{i+1}. {t.title}")
            
    except Exception as e:
        print(f"Failed to fetch discussions: {e}")

if __name__ == "__main__":
    scrape_discussions()
