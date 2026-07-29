import os
import tarfile
import shutil

def package_submission():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, "src")
    ckpt_path = os.path.join(base_dir, "checkpoints", "BC_CONVERGENCE_SWEEP.pt")
    
    tar_path = os.path.join(base_dir, "submission.tar.gz")
    
    # Files to pack, placing them at the ROOT of the tarball
    files_to_pack = {
        os.path.join(src_dir, "main.py"): "main.py",
        os.path.join(src_dir, "model.py"): "model.py",
        ckpt_path: "TOP_ELO_BC_MODEL_FINAL.pt",  # keep same arcname so main.py loads it correctly
        os.path.join(base_dir, "deck.csv"): "deck.csv"
    }
    
    # Also pack the cg folder!
    cg_dir = os.path.join(base_dir, "data", "sample_submission", "sample_submission", "cg")
    
    print("--- Kaggle Final Submission Packaging ---")
    for fpath, arcname in files_to_pack.items():
        if not os.path.exists(fpath):
            print(f"ERROR: Could not find {fpath}")
            return
            
    if not os.path.exists(cg_dir):
        print(f"ERROR: Could not find cg directory at {cg_dir}")
        return
            
    with tarfile.open(tar_path, "w:gz") as tar:
        for fpath, arcname in files_to_pack.items():
            print(f"Adding {fpath} as {arcname}")
            tar.add(fpath, arcname=arcname)
            
        print(f"Adding cg directory as cg/")
        tar.add(cg_dir, arcname="cg")
            
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"\nCreated {tar_path}")
    print(f"File size: {size_mb:.2f} MiB")
    
    if size_mb > 100.0:
        print("WARNING: Tarball exceeds Kaggle's 100 MiB limit!")
    else:
        print("STATUS: Tarball size is under 100 MiB limit.")
    
    print("\nContents of submission.tar.gz (verifying root placement):")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            print(f" - {member.name} ({member.size} bytes)")

if __name__ == "__main__":
    package_submission()
