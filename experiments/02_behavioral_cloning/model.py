"""
TitanTransformer -- Phase 2: Behavioral Cloning
================================================
Architecture:
  1. EmbeddingBag encodes 24 sparse encoder words  -> [B, 24, D]
  2. Learned positional embeddings injected        -> [B, 24, D]
  3. TransformerEncoder processes sequence         -> [B, 24, D]
  4. Mean-pool + linear projection                 -> query [B, D]
  5. EmbeddingBag encodes N sparse action words    -> [B, N, D]
  6. Dot-product query x actions + mask            -> logits [B, N]

Input format (matches real shard / collate output):
    enc_indices     : LongTensor  [B, max_enc_tokens]    -- zero-padded
    enc_values      : FloatTensor [B, max_enc_tokens]    -- zero-padded (pad=0.0)
    enc_offsets     : LongTensor  [B, 24]                -- per-sample bag boundaries
                      (relative to each sample's token row, starting from 0)
    decoder_inputs  : list[list[tuple[LongTensor, FloatTensor]]]
                      outer len = B, inner len = n_real_actions_for_b
                      each tuple: (indices_1d, values_1d) for one action bag
    action_mask     : BoolTensor  [B, max_N]             -- True = invalid/padded slot
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Add baseline agent to path so we can import encoder_size / decoder_size
# ---------------------------------------------------------------------------
_baseline_agent = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../01_baseline/agent")
)
if _baseline_agent not in sys.path:
    sys.path.insert(0, _baseline_agent)

from parser import encoder_size, decoder_size  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TitanConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_actions: int = 64   # maximum number of candidate actions per step
    n_words: int = 24       # fixed encoder sequence length (always 24)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TitanTransformer(nn.Module):
    """
    Transformer-based policy network for Pokemon TCG.

    Forward inputs (match real shard collate output):
        enc_indices    : LongTensor  [B, max_enc_tokens]
                         Padded with zeros. Weight=0.0 for padding positions.
        enc_values     : FloatTensor [B, max_enc_tokens]
                         Per-token bag weights; 0.0 for padding.
        enc_offsets    : LongTensor  [B, 24]
                         Bag start positions relative to each sample row (0-indexed).
        decoder_inputs : list[list[tuple[LongTensor, FloatTensor]]]
                         Outer: batch (len B). Inner: real actions for that sample.
                         Each tuple: (1D indices, 1D values) for one action bag.
        action_mask    : BoolTensor  [B, max_N]
                         True = invalid / padded action slot.

    Returns:
        logits : FloatTensor [B, max_N]
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # ---- Embeddings ------------------------------------------------
        self.encoder_embed = nn.EmbeddingBag(
            encoder_size, d, mode="sum", sparse=False
        )
        self.decoder_embed = nn.EmbeddingBag(
            decoder_size, d, mode="sum", sparse=False
        )

        # Learned positional embeddings for the 24 encoder words
        self.pos_embed = nn.Embedding(config.n_words, d)

        # ---- Transformer Encoder ---------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        # ---- Query projection -----------------------------------------
        self.query_proj = nn.Linear(d, d)

        # ---- Weight init ----------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.normal_(self.encoder_embed.weight, std=0.02)
        nn.init.normal_(self.decoder_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    def _embed_encoder(
        self,
        enc_indices: torch.Tensor,   # [B, max_tokens]
        enc_values:  torch.Tensor,   # [B, max_tokens]
        enc_offsets: torch.Tensor,   # [B, 24]
    ) -> torch.Tensor:               # [B, 24, D]
        """
        Embed the 24-word encoder sequence for a batch.

        Strategy: flatten [B, max_tokens] -> [B*max_tokens] and shift each
        sample's offsets by (b * max_tokens) so that EmbeddingBag receives
        monotonically non-decreasing global offsets.

        Padding tokens (weight=0.0) contribute zero to the bag sum, so
        zero-padded positions are invisible to the embedding.
        """
        B, max_tokens = enc_indices.shape

        # Flatten across batch
        flat_idx = enc_indices.reshape(-1)          # [B * max_tokens]
        flat_val = enc_values.reshape(-1)           # [B * max_tokens]

        # Shift per-sample relative offsets to global positions
        # enc_offsets[b, w] + b * max_tokens
        shift = (
            torch.arange(B, device=enc_indices.device).unsqueeze(1) * max_tokens
        )                                           # [B, 1]
        global_offsets = (enc_offsets + shift).reshape(-1)  # [B*24]

        enc_emb = self.encoder_embed(
            flat_idx, global_offsets, per_sample_weights=flat_val
        )                                           # [B*24, D]
        return enc_emb.view(B, self.config.n_words, self.config.d_model)  # [B, 24, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        enc_indices:    torch.Tensor,              # [B, max_enc_tokens]
        enc_values:     torch.Tensor,              # [B, max_enc_tokens]
        enc_offsets:    torch.Tensor,              # [B, 24]
        dec_indices:    torch.Tensor,
        dec_values:     torch.Tensor,
        dec_offsets:    torch.Tensor,
        action_mask:    torch.Tensor,              # [B, max_N] bool
    ) -> torch.Tensor:                             # [B, max_N]
        cfg = self.config

        # ----------------------------------------------------------------
        # 1. Embed encoder -> [B, 24, D]
        # ----------------------------------------------------------------
        enc_emb = self._embed_encoder(enc_indices, enc_values, enc_offsets)

        # ----------------------------------------------------------------
        # 2. Add learned positional embeddings
        # ----------------------------------------------------------------
        positions = torch.arange(cfg.n_words, device=enc_emb.device)
        enc_emb = enc_emb + self.pos_embed(positions).unsqueeze(0)  # [B, 24, D]

        # ----------------------------------------------------------------
        # 3. Transformer Encoder -> [B, 24, D]
        # ----------------------------------------------------------------
        ctx = self.transformer(enc_emb)                             # [B, 24, D]

        # ----------------------------------------------------------------
        # 4. Mean-pool -> board query vector [B, D]
        # ----------------------------------------------------------------
        query = ctx.mean(dim=1)                                     # [B, D]
        query = self.query_proj(query)                              # [B, D]

        # ----------------------------------------------------------------
        # 5. Embed decoder actions -> [B, max_N, D]
        # ----------------------------------------------------------------
        B = enc_indices.shape[0]
        max_N = action_mask.shape[1]

        if dec_indices.numel() == 0 or (dec_indices.numel() == 1 and dec_indices[0] == 0 and dec_values[0] == 0):
            dec_emb = torch.zeros(B * max_N, self.config.d_model, device=enc_indices.device)
        else:
            dec_emb = self.decoder_embed(dec_indices, dec_offsets, per_sample_weights=dec_values)
        
        dec_emb = dec_emb.view(B, max_N, self.config.d_model)

        # ----------------------------------------------------------------
        # 6. Dot-product: query [B,D] x actions [B,max_N,D] -> [B, max_N]
        # ----------------------------------------------------------------
        query_col = query.unsqueeze(2)                               # [B, D, 1]
        logits = torch.bmm(dec_emb, query_col).squeeze(2)           # [B, max_N]

        # ----------------------------------------------------------------
        # 7. Mask invalid action slots -> -1e9
        # ----------------------------------------------------------------
        logits = logits.masked_fill(action_mask, -1e9)

        return logits


# ---------------------------------------------------------------------------
# Quick sanity: print param count when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = TitanConfig()
    model = TitanTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TitanTransformer  |  encoder_size={encoder_size:,}  decoder_size={decoder_size:,}")
    print(f"Parameters: {n_params:,}")
