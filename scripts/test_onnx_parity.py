import torch
import onnxruntime as ort
import numpy as np
import sys
sys.path.append('src')
from model import PokemonActorCritic

def test_parity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PokemonActorCritic(num_card_types=3000, emb_dim=32, d_model=128, nhead=4, num_layers=3, num_actions=500)
    model.load_state_dict(torch.load('checkpoints/TITAN_TRANSFORMER_LEAGUE_01.pt', map_location=device))
    model.to(device)
    model.eval()
    
    ort_sess = ort.InferenceSession('checkpoints/titan_model.onnx')
    
    dummy_input_np = np.zeros((1, 120), dtype=np.float32)
    dummy_input_torch = torch.tensor(dummy_input_np).to(device)
    
    with torch.no_grad():
        pt_policy, pt_value = model(dummy_input_torch)
        pt_policy_np = pt_policy.cpu().numpy()
        
    onnx_outputs = ort_sess.run(None, {'state': dummy_input_np})
    onnx_policy_np = onnx_outputs[0]
    
    print("PyTorch Logits (First 5):", pt_policy_np[0][:5])
    print("ONNX Logits (First 5):", onnx_policy_np[0][:5])

if __name__ == "__main__":
    test_parity()
