"""
Behavioral Cloning Training Engine — Phase 2
=============================================
Trains TitanTransformer via cross-entropy loss on (state, action) pairs
extracted from human replay data.

Usage:
    python experiments/02_behavioral_cloning/train_bc.py
"""

import sys
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_baseline_agent = os.path.abspath(os.path.join(_this_dir, "../01_baseline/agent"))
if _baseline_agent not in sys.path:
    sys.path.insert(0, _baseline_agent)

from model import TitanTransformer, TitanConfig


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class BCTrainingConfig:
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 20
    checkpoint_dir: str = os.path.join(_this_dir, "checkpoints")
    log_every: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """
    Wraps pre-processed shards produced by a replay parser.

    Each shard item is a dict with keys matching the preprocessor output:
        enc_indices  : LongTensor   [n_enc_tokens]   -- 1D, variable length
        enc_values   : FloatTensor  [n_enc_tokens]   -- per-token weights
        enc_offsets  : LongTensor   [24]             -- bag boundaries (relative, 0-indexed)
        decoder_inputs : list[tuple(LongTensor, FloatTensor)]
                         one tuple per legal action: (1D idx, 1D val)
        target       : int or LongTensor scalar      -- chosen action index
    """

    def __init__(self, shard_paths: list):
        self.shard_paths = shard_paths
        cache_path = Path(shard_paths[0]).parent.parent / "shard_sizes_cache.pt"
        if cache_path.exists():
            self.shard_sizes, self.cumulative, self.total = torch.load(cache_path, weights_only=True)
            print(f"Loaded shard sizes from cache. Total items: {self.total}")
        else:
            self.shard_sizes = []
            self.cumulative = []
            cumsum = 0
            from tqdm import tqdm
            for path in tqdm(shard_paths, desc="Scanning Shard Sizes", unit="shard"):
                shard = torch.load(path, weights_only=True)
                size = len(shard)
                self.shard_sizes.append(size)
                self.cumulative.append(cumsum)
                cumsum += size
            self.total = cumsum
            torch.save((self.shard_sizes, self.cumulative, self.total), cache_path)
            print(f"Finished scanning {self.total} items. Cache saved.")
            
        self.cached_shard_idx = None
        self.cached_shard = None

    def __len__(self):
        return self.total

    def _locate(self, global_idx: int):
        lo, hi = 0, len(self.shard_paths) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cumulative[mid] <= global_idx:
                lo = mid
            else:
                hi = mid - 1
        shard_idx = lo
        local_idx = global_idx - self.cumulative[shard_idx]
        return shard_idx, local_idx

    def __getitem__(self, idx):
        shard_idx, local_idx = self._locate(idx)
        if self.cached_shard_idx != shard_idx:
            self.cached_shard = torch.load(self.shard_paths[shard_idx], weights_only=True)
            self.cached_shard_idx = shard_idx
        return self.cached_shard[local_idx]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def bc_collate(batch: list, max_actions: int = 64) -> dict:
    """
    Collates a batch of variable-length sparse samples into the format
    expected by TitanTransformer.forward().

    Encoder output:
        enc_indices : [B, max_enc_tokens]  -- zero-padded 2D LongTensor
        enc_values  : [B, max_enc_tokens]  -- zero-padded 2D FloatTensor
        enc_offsets : [B, 24]              -- per-sample relative bag starts

    Decoder output:
        decoder_inputs : list[list[tuple(LongTensor, FloatTensor)]]
                         outer len=B, inner len=n_real_actions for that sample

    Other:
        action_mask : [B, max_actions]  BoolTensor (True = invalid)
        target      : [B]               LongTensor
    """
    B = len(batch)

    # ---- Encoder: zero-pad to max_enc_tokens per sample ---------------
    enc_lengths = [len(item["encoder_indices"]) for item in batch]
    max_enc = max(enc_lengths)

    enc_indices_2d = torch.zeros(B, max_enc, dtype=torch.long)
    enc_values_2d  = torch.zeros(B, max_enc, dtype=torch.float32)
    enc_offsets_2d = torch.zeros(B, 24,      dtype=torch.long)

    for b, item in enumerate(batch):
        n = enc_lengths[b]
        enc_indices_2d[b, :n] = item["encoder_indices"]
        enc_values_2d[b,  :n] = item["encoder_values"]
        enc_offsets_2d[b]     = item["encoder_offsets"]   # already [24], relative

    # ---- Flatten decoder inputs across [B, max_N] into 1D tensors ---
    # action_mask communicates to the model which slots are valid.
    masks = []
    targets = []
    
    all_dec_idx = []
    all_dec_val = []
    dec_offsets = []
    cursor = 0

    for item in batch:
        dec_in = item["decoder_inputs"]   # list[tuple(LongTensor, FloatTensor)]
        n_real = len(dec_in)
        for a in range(max_actions):
            dec_offsets.append(cursor)
            if a < n_real:
                idx, val = dec_in[a][:2]
                all_dec_idx.append(idx)
                all_dec_val.append(val)
                cursor += idx.numel()

        mask = torch.ones(max_actions, dtype=torch.bool)
        mask[:n_real] = False
        masks.append(mask)

        tgt = item["target"]
        targets.append(tgt if isinstance(tgt, torch.Tensor) else torch.tensor(tgt, dtype=torch.long))

    dec_indices = torch.cat(all_dec_idx) if all_dec_idx else torch.zeros(1, dtype=torch.long)
    dec_values = torch.cat(all_dec_val) if all_dec_val else torch.zeros(1, dtype=torch.float32)
    dec_offsets_t = torch.tensor(dec_offsets, dtype=torch.long)

    return {
        "enc_indices":    enc_indices_2d,
        "enc_values":     enc_values_2d,
        "enc_offsets":    enc_offsets_2d,
        "dec_indices":    dec_indices,
        "dec_values":     dec_values,
        "dec_offsets":    dec_offsets_t,
        "action_mask":    torch.stack(masks),
        "target":         torch.stack(targets),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ShardSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        import random
        shard_indices = list(range(len(self.dataset.shard_paths)))
        random.shuffle(shard_indices)
        
        batch = []
        for shard_idx in shard_indices:
            start_idx = self.dataset.cumulative[shard_idx]
            end_idx = start_idx + self.dataset.shard_sizes[shard_idx]
            
            items_in_shard = list(range(start_idx, end_idx))
            random.shuffle(items_in_shard)
            
            for idx in items_in_shard:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

class BehavioralCloningTrainer:
    def __init__(
        self,
        config: BCTrainingConfig,
        model: TitanTransformer,
        dataset: BCDataset,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.dataset = dataset

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self.loader = DataLoader(
            dataset,
            batch_sampler=ShardSampler(dataset, config.batch_size),
            collate_fn=bc_collate,
            num_workers=0,
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        self.criterion = nn.CrossEntropyLoss()

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> dict:
        """Move tensor fields to device; skip decoder_inputs (nested list of tensors)."""
        d = self.config.device
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(d)
            else:
                result[k] = v
        return result

    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        total_correct_top1 = 0
        total_correct_top3 = 0
        total_samples = 0
        step = 0

        from tqdm import tqdm
        pbar = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)

            logits = self.model(
                batch["enc_indices"],
                batch["enc_values"],
                batch["enc_offsets"],
                batch["dec_indices"],
                batch["dec_values"],
                batch["dec_offsets"],
                batch["action_mask"],
            )

            loss = self.criterion(logits, batch["target"])

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            correct_top1 = (preds == batch["target"]).sum().item()
            
            _, top3_preds = logits.topk(min(3, logits.size(-1)), dim=-1)
            correct_top3 = (top3_preds == batch["target"].unsqueeze(1)).sum().item()

            B = batch["target"].shape[0]
            total_loss += loss.item() * B
            total_correct_top1 += correct_top1
            total_correct_top3 += correct_top3
            total_samples += B
            step += 1

            if step % self.config.log_every == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "top1": f"{correct_top1/B:.3f}",
                    "top3": f"{correct_top3/B:.3f}"
                })

        self.scheduler.step()

        return {
            "loss": total_loss / max(total_samples, 1),
            "top1_acc": total_correct_top1 / max(total_samples, 1),
            "top3_acc": total_correct_top3 / max(total_samples, 1),
        }

    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, metrics: dict):
        path = Path(self.config.checkpoint_dir) / f"titan_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        print(f"  Checkpoint saved: {path}")

    # ------------------------------------------------------------------

    def run(self):
        cfg = self.config
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"  TitanTransformer BC Training")
        print(f"  Device      : {cfg.device}")
        print(f"  Parameters  : {n_params:,}")
        print(f"  Dataset size: {len(self.dataset):,}")
        print(f"  Epochs      : {cfg.epochs}")
        print(f"  Batch size  : {cfg.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(epoch)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3d}/{cfg.epochs}  "
                f"loss={metrics['loss']:.4f}  "
                f"top1_acc={metrics['top1_acc']:.4f}  "
                f"top3_acc={metrics['top3_acc']:.4f}  "
                f"({elapsed:.1f}s)"
            )
            self.save_checkpoint(epoch, metrics)
            if metrics["loss"] < self.best_loss:
                self.best_loss = metrics["loss"]
                best_path = Path(self.config.checkpoint_dir) / "titan_bc_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    best_path,
                )
                print(f"  Best checkpoint saved: {best_path}")

        print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    import multiprocessing
    multiprocessing.freeze_support()

    shard_dir = os.path.join(_this_dir, "shards")
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))

    if not shard_paths:
        print(
            "No shards found. Run the replay → shard converter first.\n"
            f"Expected location: {shard_dir}/*.pt"
        )
        sys.exit(1)

    train_config = BCTrainingConfig()
    titan_config = TitanConfig()

    dataset = BCDataset(shard_paths)
    model = TitanTransformer(titan_config)

    trainer = BehavioralCloningTrainer(train_config, model, dataset)
    trainer.run()
