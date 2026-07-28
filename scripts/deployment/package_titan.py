import tarfile
import os

print("Packaging TITAN True Submission...")
with tarfile.open("submission.tar.gz", "w:gz") as tar:
    tar.add("main.py")
    tar.add("deck.csv")
    tar.add("titan_model.onnx")
    tar.add("src/kaggle_libs/onnxruntime_pkg", arcname="src/kaggle_libs/onnxruntime_pkg")

print(f"Done. Byte Size: {os.path.getsize('submission.tar.gz')}")
