import sys
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

_baseline_agent = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../01_baseline/agent")
)
if _baseline_agent not in sys.path:
    sys.path.insert(0, _baseline_agent)

from parser import encoder_size, decoder_size

@dataclass
class TitanConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_actions: int = 64
    n_words: int = 24

class TitanTransformerPPO(nn.Module):
    def __init__(self, config: TitanConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.encoder_embed = nn.EmbeddingBag(encoder_size, d, mode="sum", sparse=False)
        self.decoder_embed = nn.EmbeddingBag(decoder_size, d, mode="sum", sparse=False)
        self.pos_embed = nn.Embedding(config.n_words, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_heads, dim_feedforward=config.d_ff, dropout=config.dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.query_proj = nn.Linear(d, d)

        self.critic = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.encoder_embed.weight, std=0.02)
        nn.init.normal_(self.decoder_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _embed_encoder(self, enc_indices, enc_values, enc_offsets):
        B, max_tokens = enc_indices.shape
        flat_idx = enc_indices.reshape(-1)
        flat_val = enc_values.reshape(-1)
        shift = (torch.arange(B, device=enc_indices.device).unsqueeze(1) * max_tokens)
        global_offsets = (enc_offsets + shift).reshape(-1)
        enc_emb = self.encoder_embed(flat_idx, global_offsets, per_sample_weights=flat_val)
        return enc_emb.view(B, self.config.n_words, self.config.d_model)

    def forward(self, enc_indices, enc_values, enc_offsets, dec_indices, dec_values, dec_offsets, action_mask):
        cfg = self.config
        enc_emb = self._embed_encoder(enc_indices, enc_values, enc_offsets)
        positions = torch.arange(cfg.n_words, device=enc_emb.device)
        enc_emb = enc_emb + self.pos_embed(positions).unsqueeze(0)
        ctx = self.transformer(enc_emb)
        query = ctx.mean(dim=1)
        
        value = self.critic(query)

        query = self.query_proj(query)
        B = enc_indices.shape[0]
        max_N = action_mask.shape[1]

        if dec_indices.numel() == 0 or (dec_indices.numel() == 1 and dec_indices[0] == 0 and dec_values[0] == 0):
            dec_emb = torch.zeros(B * max_N, self.config.d_model, device=enc_indices.device)
        else:
            dec_emb = self.decoder_embed(dec_indices, dec_offsets, per_sample_weights=dec_values)
        
        dec_emb = dec_emb.view(B, max_N, self.config.d_model)

        query_col = query.unsqueeze(2)
        logits = torch.bmm(dec_emb, query_col).squeeze(2)
        
        logits = logits.masked_fill(action_mask, -1e9)

        return logits, value
