import os
import json
import torch
import torch.nn as nn
from agent.parser import (
    get_encoder_input,
    get_decoder_input,
    SparseVector,
    card_count,
    attack_count,
    decoder_size
)
from agent.cg.api import to_observation_class

# Constants
num_words_encoder = 24
encoder_size = 22000 # Encoder input size exceeding the maximum possible vocabulary size

class DecoderLayer(nn.Module):
    """
    A custom Decoder Layer that ONLY uses Cross-Attention.
    Self-Attention is disabled because actions are independent and don't need to attend to each other.
    """
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super(DecoderLayer, self).__init__()
        # Cross Attention: query is action, key/value are encoder_out (board state)
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, d_feedforward)
        self.fc2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # Cross Attention: Action queries the Board State
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)

class MyModel(nn.Module):
    """
    Actor-Critic Transformer for Pokemon TCG.
    Takes Encoder SparseVector (Board) and Decoder SparseVector (Actions).
    Outputs (Value, Policy).
    """
    def __init__(self,
                 d_model: int = 128,
                 num_heads: int = 4,
                 d_feedforward: int = 256,
                 num_layers_encoder: int = 2,
                 num_layers_decoder: int = 2):
        super(MyModel, self).__init__()

        self.d_model = d_model

        # 1. ENCODER (Critic / Board State)
        self.encoder_bag = nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0) # 0 dropout
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = nn.Linear(d_model, 1)

        # 2. DECODER (Actor / Action Policy)
        self.decoder_bag = nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = nn.Linear(d_model, 1)

    def forward(self,
                index_encoder: torch.Tensor,
                value_encoder: torch.Tensor,
                offset_encoder: torch.Tensor,
                index_decoder: torch.Tensor,
                value_decoder: torch.Tensor,
                offset_decoder: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # --- Process Board State (Encoder) ---
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        # Reshape to [24, batch_size, d_model] for Transformer
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        
        # Self-Attention across the 24 words
        encoder_out = self.encoder(v)
        
        # Critic Value Head: Average the 24 encoded words into a single score
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        # --- Process Actions (Decoder) ---
        # Bag the actions
        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        
        # Reshape to [N_actions, batch_size, d_model]
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        
        # Cross-Attention against the Board State
        for layer in self.decoder:
            p = layer(p, encoder_out)
            
        # Actor Policy Head: Score each action
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        
        # We don't tanh the policy output if we use it as logits for Categorical sampling,
        # but the baseline author used tanh for some reason. We will stick to the baseline for now.
        # Actually, in PPO, policy logits are usually unconstrained. But we will use the notebook's exact design.
        p = torch.tanh(p)
        
        return (v, p)


if __name__ == '__main__':
    print("Loading replay_20260730_125951.json...")
    replay_path = os.path.join(os.path.dirname(__file__), '..', 'replay_20260730_125951.json')
    with open(replay_path, 'r', encoding='utf-8') as f:
        replay = json.load(f)
        
    step_data = replay['steps'][10]
    obs_dict = step_data[0]['observation']
    obs = to_observation_class(obs_dict)
    
    # 1. Parse Board State (Encoder)
    sv_enc = get_encoder_input(obs)
    enc_idx = torch.tensor(sv_enc.index, dtype=torch.int32)
    enc_val = torch.tensor(sv_enc.value, dtype=torch.float32)
    enc_off = torch.tensor(sv_enc.offset, dtype=torch.int32)
    
    # 2. Parse Actions (Decoder)
    legal_option_count = len(obs.select.option)
    actions = [[i] for i in range(legal_option_count)]
    sv_dec = get_decoder_input(obs, actions)
    dec_idx = torch.tensor(sv_dec.index, dtype=torch.int32)
    dec_val = torch.tensor(sv_dec.value, dtype=torch.float32)
    dec_off = torch.tensor(sv_dec.offset, dtype=torch.int32)
    
    # 3. Initialize Model
    print("Initializing MyModel (Actor-Critic Transformer)...")
    model = MyModel()
    
    # 4. Forward Pass
    print(f"\n--- RUNNING FORWARD PASS ---")
    print(f"Board Words: 24")
    print(f"Legal Action Words: {legal_option_count}")
    
    value, policy = model(
        index_encoder=enc_idx, value_encoder=enc_val, offset_encoder=enc_off,
        index_decoder=dec_idx, value_decoder=dec_val, offset_decoder=dec_off
    )
    
    print("\n--- FORWARD PASS SUCCESS ---")
    print(f"Value Tensor Shape:  {value.shape}  -> Expected: [1, 1]")
    print(f"Policy Tensor Shape: {policy.shape}  -> Expected: [1, {legal_option_count}]")
    
    assert value.shape == (1, 1), f"Value shape incorrect! Got {value.shape}"
    assert policy.shape == (1, legal_option_count), f"Policy shape incorrect! Got {policy.shape}"
    
    print(f"\nValue Score:  {value.item():.4f}")
    print(f"Policy Scores: {policy.detach().numpy()[0]}")
    print("\nMATHEMATICAL PROOF: The network successfully evaluates the board to exactly 1 score (Value) and maps each of the available legal actions to exactly 1 score (Policy)!")
