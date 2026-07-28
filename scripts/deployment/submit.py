import os

def submit():
    print("Submitting to Kaggle Simulation Track...")
    cmd = "kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz -m \"TITAN V5.0 - Transformer League 1200+ BC\""
    exit_code = os.system(cmd)
    if exit_code == 0:
        print("Successfully submitted to Kaggle!")
    else:
        print("Submission failed or simulated.")

if __name__ == "__main__":
    submit()
