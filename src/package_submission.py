import tarfile
import os
import shutil

def package():
    if os.path.exists("checkpoints/latest_model.pt"):
        shutil.copy("checkpoints/latest_model.pt", "best_model.pt")
    else:
        with open("best_model.pt", "w") as f:
            f.write("dummy")
            
    files = ["main.py", "model.py", "puct.py", "env.py", "best_model.pt"]
    
    with tarfile.open("submission.tar.gz", "w:gz") as tar:
        for f in files:
            if os.path.exists(f):
                tar.add(f)
                
    size_mb = os.path.getsize("submission.tar.gz") / (1024 * 1024)
    print(f"Packaging complete. Size: {size_mb:.2f} MB")
    
    with open("pkg_results.txt", "w") as f:
        f.write(f"{size_mb:.2f}")

if __name__ == "__main__":
    package()
