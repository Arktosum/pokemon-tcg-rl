import os
import subprocess
import glob
import zipfile
import shutil

def download():
    print("Downloading ONNX runtime for manylinux...")
    target_dir = "./kaggle_libs"
    pkg_dir = os.path.join(target_dir, "onnxruntime_pkg")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pkg_dir, exist_ok=True)
    
    cmd = [
        "pip", "download", "onnxruntime", 
        "--platform", "manylinux2014_x86_64", 
        "--only-binary=:all:", 
        "-d", target_dir
    ]
    subprocess.run(cmd, check=True)
    
    wheels = glob.glob(os.path.join(target_dir, "*.whl"))
    for wheel in wheels:
        print(f"Extracting {wheel}...")
        with zipfile.ZipFile(wheel, 'r') as zip_ref:
            zip_ref.extractall(pkg_dir)
        os.remove(wheel)
        
    print("Dependencies smuggled successfully.")

if __name__ == "__main__":
    download()
