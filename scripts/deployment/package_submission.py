import tarfile
import os

def package():
    print("Packaging submission...")
    with tarfile.open("submission.tar.gz", "w:gz") as tar:
        tar.add("main.py")
        if os.path.exists("checkpoints/titan_model.onnx"):
            tar.add("checkpoints/titan_model.onnx", arcname="titan_model.onnx")
        if os.path.exists("kaggle_libs"):
            tar.add("kaggle_libs")
    
    size = os.path.getsize("submission.tar.gz") / (1024 * 1024)
    print(f"submission.tar.gz created. Size: {size:.2f} MB")

if __name__ == "__main__":
    package()
