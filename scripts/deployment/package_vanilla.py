import tarfile
import os

print("Packaging Vanilla Submission...")
with tarfile.open("submission_vanilla.tar.gz", "w:gz") as tar:
    tar.add("main.py")
    tar.add("deck.csv")

print(f"Done. Byte Size: {os.path.getsize('submission_vanilla.tar.gz')}")
