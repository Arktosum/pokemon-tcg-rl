import torch
import torch.nn as nn
import torch.nn.functional as F

class PokemonActorCritic(nn.Module):
    def __init__(self, num_card_types=3000, emb_dim=32, d_model=128, nhead=4, num_layers=3, num_actions=500):
        super().__init__()
        self.d_model = d_model
        
        self.card_embedding = nn.Embedding(num_card_types, emb_dim)
        
        # Projectors for different token types
        self.global_proj = nn.Linear(4, d_model)
        self.stats_proj = nn.Linear(9, d_model)
        self.pk_proj = nn.Linear(emb_dim + 3, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.policy_head = nn.Linear(d_model, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def _process_pk_token(self, pk_tensor):
        # pk_tensor: (B, 4) - ID, HP, MaxHP, Energy
        card_id = pk_tensor[:, 0].long()
        stats = pk_tensor[:, 1:]
        emb = self.card_embedding(card_id) # (B, emb_dim)
        combined = torch.cat([emb, stats], dim=-1) # (B, emb_dim + 3)
        return self.pk_proj(combined).unsqueeze(1) # (B, 1, d_model)
        
    def forward(self, state, action_mask=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
                
        B = state.shape[0]
        
        # 0: Global (4)
        global_tok = self.global_proj(state[:, 0:4]).unsqueeze(1) # (B, 1, d_model)
        
        # 1: P0 Stats (9)
        p0_stats_tok = self.stats_proj(state[:, 4:13]).unsqueeze(1) # (B, 1, d_model)
        
        # 2-7: P0 PKs
        p0_pks = []
        for i in range(6):
            idx = 13 + i * 4
            p0_pks.append(self._process_pk_token(state[:, idx:idx+4]))
            
        # 8: P1 Stats (9)
        p1_stats_tok = self.stats_proj(state[:, 37:46]).unsqueeze(1) # (B, 1, d_model)
        
        # 9-14: P1 PKs
        p1_pks = []
        for i in range(6):
            idx = 46 + i * 4
            p1_pks.append(self._process_pk_token(state[:, idx:idx+4]))
            
        tokens = [global_tok, p0_stats_tok] + p0_pks + [p1_stats_tok] + p1_pks
        seq = torch.cat(tokens, dim=1) # (B, 15, d_model)
        
        # Transformer pass
        out_seq = self.transformer(seq) # (B, 15, d_model)
        
        # Pool (mean pooling)
        pooled = out_seq.mean(dim=1) # (B, d_model)
        pooled = self.layer_norm(pooled)
        
        logits = self.policy_head(pooled)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
            
        policy_probs = F.softmax(logits, dim=-1)
        value = self.value_head(pooled)
        
        return policy_probs, value
