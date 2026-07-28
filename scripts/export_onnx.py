import os
import sys
import torch
import torch.onnx

sys.path.append('src')
from model import PokemonActorCritic

def export():
    print("Exporting model to ONNX...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PokemonActorCritic(num_card_types=3000, emb_dim=32, d_model=128, nhead=4, num_layers=3, num_actions=500)
    
    model_path = os.path.join('checkpoints', 'TITAN_TRANSFORMER_LEAGUE_01.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    dummy_input = torch.zeros(1, 120).to(device)
    
    onnx_path = os.path.join('checkpoints', 'titan_model.onnx')
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True,
        input_names=['state'], 
        output_names=['policy_probs', 'value'],
        dynamic_axes={'state': {0: 'batch_size'}, 'policy_probs': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
    )
    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    export()
