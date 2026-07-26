import os
import sys
import shutil
import tarfile
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_submission():
    base_dir = r"g:\programming\github-repositories\pokemon-tcg-rl"
    submission_dir = os.path.join(base_dir, "submission")
    
    # 1. Clean and Create Directory
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)
    logger.info("Created submission directory.")
    
    # 2. Find and Copy Latest Checkpoint
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    checkpoints = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    if not checkpoints:
        logger.error("No checkpoints found to package!")
        sys.exit(1)
        
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    dest_ckpt = os.path.join(submission_dir, "model.pt")
    shutil.copy2(latest_ckpt, dest_ckpt)
    logger.info(f"Copied latest checkpoint: {latest_ckpt} -> {dest_ckpt}")
    
    # 3. Copy Transformer Policy
    src_policy = os.path.join(base_dir, "src", "model", "transformer_policy.py")
    dest_policy = os.path.join(submission_dir, "transformer_policy.py")
    shutil.copy2(src_policy, dest_policy)
    logger.info(f"Copied policy script: {src_policy} -> {dest_policy}")
    
    # 4. Generate Deck CSV (using Roman Rozen's standard deck as a placeholder)
    deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    deck_path = os.path.join(submission_dir, "deck.csv")
    with open(deck_path, "w") as f:
        f.write("\n".join(map(str, deck)))
    logger.info("Generated deck.csv")
    
    # 5. Write main.py
    main_code = """import os
import sys
import torch
from cg.api import Observation, to_observation_class

# Add current directory to path for local module imports on Kaggle
sys.path.append(os.path.dirname(__file__))
from transformer_policy import MyModel, get_encoder_input, get_decoder_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
deck_list = None

def load_model():
    global model, deck_list
    if model is None:
        model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1).to(device)
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        deck_list = read_deck_csv()

def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().strip().split("\\n")
    deck = [int(x) for x in csv[:60]]
    return deck

def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    
    if obs.select is None:
        return read_deck_csv()
        
    load_model()
    
    sv_enc = get_encoder_input(obs, deck_list)
    legal_actions = [[i] for i in range(len(obs.select.option))]
    sv_dec = get_decoder_input(obs, legal_actions)
    
    enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32, device=device)
    enc_val = torch.tensor(sv_enc.value, dtype=torch.float32, device=device)
    enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32, device=device)
    dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32, device=device)
    dec_val = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
    dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
    
    with torch.no_grad():
        _, logits = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
        action_idx = torch.argmax(logits).item()
        
    return legal_actions[action_idx]
"""
    main_path = os.path.join(submission_dir, "main.py")
    with open(main_path, "w") as f:
        f.write(main_code)
    logger.info("Generated main.py Kaggle hook")
    
    # 6. Compress to tar.gz
    tar_path = os.path.join(base_dir, "submission.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        # Iterate through the submission directory and add files to the tar root
        for item in os.listdir(submission_dir):
            item_path = os.path.join(submission_dir, item)
            tar.add(item_path, arcname=item)
    logger.info(f"✅ Successfully packaged submission to: {tar_path}")

if __name__ == "__main__":
    create_submission()
