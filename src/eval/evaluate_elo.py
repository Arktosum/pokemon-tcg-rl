import sys
import os
import glob
import logging
import argparse
import torch

parent_cg_path = r'g:\programming\github-repositories\pokemon-tcg-rl\input\sample_submission\sample_submission'
if os.path.exists(parent_cg_path) and parent_cg_path not in sys.path:
    sys.path.append(parent_cg_path)
sys.path.append(r'g:\programming\github-repositories\pokemon-tcg-rl')

from cg.api import to_observation_class
from src.model.transformer_policy import MyModel, get_encoder_input, get_decoder_input
from src.env.tcg_env import PokemonTCGEnv
from src.model.heuristic_bot import agent as rule_agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def load_latest_checkpoint(device, checkpoint_dir='checkpoints', explicit_path=None):
    if explicit_path:
        latest_ckpt = explicit_path
        logger.info(f"Loading explicit checkpoint for evaluation: {latest_ckpt}")
    else:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if not checkpoints:
            logger.error("No checkpoints found!")
            return None
            
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        logger.info(f"Loading latest checkpoint for evaluation: {latest_ckpt}")
    
    model = MyModel(d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=1, num_layers_decoder=1).to(device)
    checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting Local ELO Tournament on {device}...")
    
    deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    env = PokemonTCGEnv(deck)
    
    agent_model = load_latest_checkpoint(device, explicit_path=args.ckpt)
    if agent_model is None:
        return
        
    matches = args.matches
    wins = 0
    draws = 0
    losses = 0
    
    logger.info(f"Opponent: Roman Rozen Heuristic Baseline Bot")
    logger.info(f"Total Matches to play: {matches}")
    
    for match in range(1, matches + 1):
        obs_dict = env.reset()
        step_count = 0
        
        while True:
            if not obs_dict:
                break
                
            your_index = obs_dict['current']['yourIndex']
            
            if your_index == 0:
                obs = to_observation_class(obs_dict)
                sv_enc = get_encoder_input(obs, env.deck)
                legal_actions = [[i] for i in range(len(obs.select.option))]
                sv_dec = get_decoder_input(obs, legal_actions)
                
                enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32, device=device)
                enc_val = torch.tensor(sv_enc.value, dtype=torch.float32, device=device)
                enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32, device=device)
                dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32, device=device)
                dec_val = torch.tensor(sv_dec.value, dtype=torch.float32, device=device)
                dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32, device=device)
                
                with torch.no_grad():
                    _, logits = agent_model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
                    action_idx = torch.argmax(logits).item() # Greedy evaluation
                
                chosen_action = legal_actions[action_idx]
                obs_dict, reward, done = env.step(chosen_action)
                
                if done:
                    if reward == 1.0: wins += 1
                    elif reward == 0.0: draws += 1
                    else: losses += 1
                    break
                    
            else:
                try:
                    chosen_action = rule_agent(obs_dict)
                except Exception:
                    wins += 1 # Bot crashed
                    break
                    
                obs_dict, reward, done = env.step(chosen_action)
                if done:
                    if reward == -1.0: losses += 1
                    elif reward == 0.0: draws += 1
                    else: wins += 1
                    break
                    
            step_count += 1
            if step_count > 500:
                draws += 1
                break
                
        win_rate = wins / match
        logger.info(f"Match {match:03d}/{matches} | Result: {'WIN' if reward == 1.0 else 'LOSS' if reward == -1.0 else 'DRAW'} | Current WR: {win_rate:.1%}")
        
    logger.info("="*40)
    logger.info("LOCAL ELO TOURNAMENT RESULTS")
    logger.info(f"Wins: {wins} | Losses: {losses} | Draws: {draws}")
    logger.info(f"Raw Win Rate: {wins/matches:.2%}")
    expected_score = (wins + (0.5 * draws)) / matches
    logger.info(f"TrueSkill Expected Score: {expected_score:.2%}")
    logger.info("="*40)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--matches', type=int, default=10, help="Number of evaluation matches")
    parser.add_argument('--ckpt', type=str, default=None, help="Explicit path to a .pt checkpoint")
    args = parser.parse_args()
    evaluate(args)
