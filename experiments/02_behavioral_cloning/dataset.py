"""
dataset.py – PokemonReplayDataset for Behavioral Cloning
=========================================================
Lazily loads .pt shard files produced by preprocess.py.

Usage:
    from dataset import PokemonReplayDataset, ExpertSample, pad_replay_batch
    from torch.utils.data import DataLoader

    ds = PokemonReplayDataset("experiments/02_behavioral_cloning/shards")
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        collate_fn=pad_replay_batch)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


# ── Data container ───────────────────────────────────────────────────────────

@dataclass
class ExpertSample:
    """A single (state, action) pair extracted from a match replay."""
    encoder_indices: torch.Tensor   # [N_enc_tokens]   – sparse indices for 24 encoder words
    encoder_values:  torch.Tensor   # [N_enc_tokens]   – corresponding float weights
    encoder_offsets: torch.Tensor   # [24]             – word boundaries (EmbeddingBag input)
    decoder_inputs:  list           # list of (indices, values, offsets) tuples, one per legal action
    target:          int            # index of the chosen action within decoder_inputs


# ── Dataset ──────────────────────────────────────────────────────────────────

class PokemonReplayDataset(Dataset):
    """
    Loads all .pt shards from a directory lazily (shard is loaded on first access
    of any sample within it).

    Each shard file is a list[dict] with keys:
        encoder_indices, encoder_values, encoder_offsets,
        decoder_inputs, target
    """

    def __init__(self, shards_dir: str | Path):
        self.shards_dir = Path(shards_dir)
        shard_paths = sorted(self.shards_dir.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard files found in {self.shards_dir}. "
                "Run preprocess.py first."
            )

        # Build a cumulative index: for each global index, which shard + local index?
        self._shard_paths:  List[Path] = shard_paths
        self._shard_sizes:  List[int]  = []
        self._cumulative:   List[int]  = []   # cumulative sum of sizes

        # We need sizes without loading all shards into memory.
        # Approach: load each shard once to count, then discard.
        # For very large datasets, consider storing sizes in a manifest file;
        # for now this is fast enough.
        cumsum = 0
        for path in shard_paths:
            shard = torch.load(path, weights_only=False)
            size  = len(shard)
            self._shard_sizes.append(size)
            self._cumulative.append(cumsum)
            cumsum += size
        self._total = cumsum

        # Lazy cache: shard index -> loaded list (LRU not implemented; keep last loaded)
        self._cached_shard_idx: int | None = None
        self._cached_shard:     list | None = None

    def __len__(self) -> int:
        return self._total

    def _get_shard(self, shard_idx: int) -> list:
        """Load and cache a shard by index."""
        if self._cached_shard_idx != shard_idx:
            self._cached_shard = torch.load(
                self._shard_paths[shard_idx], weights_only=False
            )
            self._cached_shard_idx = shard_idx
        return self._cached_shard  # type: ignore[return-value]

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        """Return (shard_idx, local_idx) for a global sample index."""
        # Binary search in cumulative sizes
        lo, hi = 0, len(self._shard_paths) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cumulative[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        shard_idx = lo
        local_idx = global_idx - self._cumulative[shard_idx]
        return shard_idx, local_idx

    def __getitem__(self, idx: int) -> ExpertSample:
        shard_idx, local_idx = self._locate(idx)
        raw = self._get_shard(shard_idx)[local_idx]
        return ExpertSample(
            encoder_indices = raw["encoder_indices"],
            encoder_values  = raw["encoder_values"],
            encoder_offsets = raw["encoder_offsets"],
            decoder_inputs  = raw["decoder_inputs"],
            target          = int(raw["target"]),
        )


# ── Collate function ─────────────────────────────────────────────────────────

def pad_replay_batch(batch: List[ExpertSample]) -> dict:
    """
    Collate function for DataLoader.

    Returns a dict with:
        encoder_indices : LongTensor  [B, max_enc_tokens]   – zero-padded
        encoder_values  : FloatTensor [B, max_enc_tokens]   – zero-padded
        encoder_offsets : LongTensor  [B, 24]               – stacked (no padding needed)
        decoder_tensor  : FloatTensor [B, max_N, decoder_dim] – *not* produced here;
                          for sparse decoder inputs use the raw lists below
        decoder_indices : list of list of (idx, val, off) tuples  – B outer, N_i inner
        action_mask     : BoolTensor  [B, max_N]            – True where PADDED (invalid)
        targets         : LongTensor  [B]

    Notes on the decoder:
    ---------------------
    The decoder uses EmbeddingBag so the raw sparse format is the most efficient
    input. We return `decoder_indices` as a nested list; the model's forward pass
    should iterate over the batch to build per-sample embeddings, then pad+stack.

    If you prefer a dense tensor for simpler batching, set `dense_decoder=True`
    when calling this function (requires building and caching a fixed-size
    EmbeddingBag separately).
    """
    B = len(batch)

    # ── Encoder padding ──────────────────────────────────────────────────────
    enc_lens = [s.encoder_indices.shape[0] for s in batch]
    max_enc  = max(enc_lens) if enc_lens else 0

    enc_idx_pad = torch.zeros(B, max_enc, dtype=torch.long)
    enc_val_pad = torch.zeros(B, max_enc, dtype=torch.float32)
    enc_off_stk = torch.stack([s.encoder_offsets for s in batch], dim=0)  # [B, 24]

    for i, s in enumerate(batch):
        L = enc_lens[i]
        enc_idx_pad[i, :L] = s.encoder_indices
        enc_val_pad[i, :L] = s.encoder_values

    # ── Action (decoder) mask ────────────────────────────────────────────────
    n_actions = [len(s.decoder_inputs) for s in batch]
    max_N     = max(n_actions) if n_actions else 0

    # action_mask: True = PADDING (invalid action slot)
    action_mask = torch.ones(B, max_N, dtype=torch.bool)
    for i, n in enumerate(n_actions):
        action_mask[i, :n] = False  # valid positions

    # ── Targets ──────────────────────────────────────────────────────────────
    targets = torch.tensor([s.target for s in batch], dtype=torch.long)

    # ── Flatten decoder inputs across [B, max_N] into 1D tensors ────────────────
    all_dec_idx = []
    all_dec_val = []
    dec_offsets = []
    cursor = 0

    for s in batch:
        for a in range(max_N):
            dec_offsets.append(cursor)
            if a < len(s.decoder_inputs):
                idx, val = s.decoder_inputs[a][:2]
                all_dec_idx.append(idx)
                all_dec_val.append(val)
                cursor += idx.numel()

    dec_indices = torch.cat(all_dec_idx) if all_dec_idx else torch.zeros(1, dtype=torch.long)
    dec_values  = torch.cat(all_dec_val) if all_dec_val else torch.zeros(1, dtype=torch.float32)
    dec_offsets_t = torch.tensor(dec_offsets, dtype=torch.long)

    return {
        "encoder_indices": enc_idx_pad,    # [B, max_enc]
        "encoder_values":  enc_val_pad,    # [B, max_enc]
        "encoder_offsets": enc_off_stk,    # [B, 24]
        "dec_indices":     dec_indices,
        "dec_values":      dec_values,
        "dec_offsets":     dec_offsets_t,
        "action_mask":     action_mask,    # [B, max_N]  True=padded
        "targets":         targets,        # [B]
    }


# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from torch.utils.data import DataLoader

    shards_dir = Path(__file__).parent / "shards"
    if not shards_dir.exists():
        print(f"No shards directory at {shards_dir}. Run preprocess.py first.")
        sys.exit(1)

    print(f"Loading dataset from {shards_dir} ...")
    ds = PokemonReplayDataset(shards_dir)
    print(f"Total samples: {len(ds)}")

    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  encoder_indices : {sample.encoder_indices.shape}")
    print(f"  encoder_values  : {sample.encoder_values.shape}")
    print(f"  encoder_offsets : {sample.encoder_offsets.shape}")
    print(f"  decoder_inputs  : {len(sample.decoder_inputs)} options")
    print(f"  target          : {sample.target}")

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=pad_replay_batch)
    batch  = next(iter(loader))
    print(f"\nBatch (size=4):")
    print(f"  encoder_indices : {batch['encoder_indices'].shape}")
    print(f"  encoder_values  : {batch['encoder_values'].shape}")
    print(f"  encoder_offsets : {batch['encoder_offsets'].shape}")
    print(f"  action_mask     : {batch['action_mask'].shape}")
    print(f"  targets         : {batch['targets']}")
    print(f"  decoder_inputs  : {len(batch['decoder_inputs'])} samples, "
          f"first has {len(batch['decoder_inputs'][0])} options")
    print("\nDataset and DataLoader OK!")
