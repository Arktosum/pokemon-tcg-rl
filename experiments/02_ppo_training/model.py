import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

baseline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_baseline'))
if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from agent.parser import encoder_size, decoder_size

class TitanTransformer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.encoder_bag = nn.EmbeddingBag(num_embeddings=encoder_size, embedding_dim=d_model, mode='sum')
        self.decoder_bag = nn.EmbeddingBag(num_embeddings=decoder_size, embedding_dim=d_model, mode='sum')
        
        self.pos_emb = nn.Embedding(24, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, enc_indices, enc_offsets, enc_weights, dec_indices, dec_offsets, dec_weights, batch_lengths=None):
        x = self.encoder_bag(enc_indices, enc_offsets, per_sample_weights=enc_weights)
        
        B = x.size(0) // 24
        x = x.view(B, 24, self.d_model)
        
        positions = torch.arange(24, device=x.device).unsqueeze(0).expand(B, 24)
        x = x * math.sqrt(self.d_model) + self.pos_emb(positions)
        
        encoded_seq = self.transformer(x)
        global_context = encoded_seq.mean(dim=1, keepdim=True)
        
        value = self.critic_head(global_context.squeeze(1))
        
        embedded_actions = self.decoder_bag(dec_indices, dec_offsets, per_sample_weights=dec_weights)
        
        if batch_lengths is None:
            logits = torch.matmul(embedded_actions, global_context.squeeze(1).transpose(0, 1)).squeeze(1)
            logits = logits / math.sqrt(self.d_model)
            return logits.unsqueeze(0), value
            
        else:
            max_N = batch_lengths.max().item()
            padded_actions = torch.zeros(B, max_N, self.d_model, device=x.device)
            mask = torch.ones(B, max_N, dtype=torch.bool, device=x.device)
            
            start_idx = 0
            for i, n in enumerate(batch_lengths):
                if n > 0:
                    padded_actions[i, :n, :] = embedded_actions[start_idx:start_idx+n, :]
                    mask[i, :n] = False
                    start_idx += n
                    
            logits = torch.bmm(global_context, padded_actions.transpose(1, 2)).squeeze(1)
            logits = logits / math.sqrt(self.d_model)
            logits = logits.masked_fill(mask, float('-inf'))
            
            return logits, value
