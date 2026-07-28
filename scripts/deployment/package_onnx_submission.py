import tarfile
import os

def package():
    print("Packaging submission...")
    with tarfile.open("submission.tar.gz", "w:gz") as tar:
        tar.add("main.py")
        tar.add("deck.csv")
        # I am moving the file from checkpoints/titan_model.onnx to the root of the archive.
        tar.add("checkpoints/titan_model.onnx", arcname="titan_model.onnx")
        
        # Include src/kaggle_libs/onnxruntime_pkg/ and src/env.py
        tar.add("src/kaggle_libs/onnxruntime_pkg", arcname="src/kaggle_libs/onnxruntime_pkg")
        if os.path.exists("src/env.py"):
            tar.add("src/env.py", arcname="src/env.py")
            
    size = os.path.getsize("submission.tar.gz")
    print(f"submission.tar.gz created successfully. Size: {size} bytes")

if __name__ == "__main__":
    package()
