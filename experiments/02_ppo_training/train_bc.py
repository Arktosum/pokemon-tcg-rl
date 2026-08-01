import gc
import sys
import time
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from model import TitanTransformer


def ts():
    """Return current timestamp string."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------
# 1. DATASET & COLLATOR
# ---------------------------------------------------------
class PokemonBCDataset(Dataset):
    def __init__(self, pt_file: str, max_samples: int = None):
        print(f"{ts()} Loading dataset from {pt_file}...")
        sys.stdout.flush()
        self.samples = torch.load(pt_file, map_location='cpu', weights_only=False)
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"{ts()} Loaded {len(self.samples)} samples.")
        sys.stdout.flush()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Dynamically collates a batch of samples into massive 1D sparse tensors
    ready for EmbeddingBag, correctly shifting offsets.
    """
    enc_indices, enc_weights, enc_offsets = [], [], []
    dec_indices, dec_weights, dec_offsets = [], [], []

    enc_idx_offset = 0
    dec_idx_offset = 0

    batch_lengths = []
    targets = []

    for sample in batch:
        ei = torch.tensor(sample['encoder_indices'], dtype=torch.int32)
        ew = torch.tensor(sample['encoder_values'], dtype=torch.float32)
        eo = torch.tensor(sample['encoder_offsets'], dtype=torch.int32)
        enc_indices.append(ei)
        enc_weights.append(ew)
        enc_offsets.append(eo + enc_idx_offset)
        enc_idx_offset += len(ei)

        di = torch.tensor(sample['decoder_indices'], dtype=torch.int32)
        dw = torch.tensor(sample['decoder_values'], dtype=torch.float32)
        do_ = torch.tensor(sample['decoder_offsets'], dtype=torch.int32) if sample['legal_option_count'] > 0 else torch.empty(0, dtype=torch.int32)
        dec_indices.append(di)
        dec_weights.append(dw)
        dec_offsets.append(do_ + dec_idx_offset)
        dec_idx_offset += len(di)

        batch_lengths.append(sample['legal_option_count'])

        target = sample['target_actions']
        targets.append(target[0] if isinstance(target, list) and len(target) > 0 else 0)

    return (
        torch.cat(enc_indices) if enc_indices else torch.empty(0, dtype=torch.int32),
        torch.cat(enc_offsets) if enc_offsets else torch.empty(0, dtype=torch.int32),
        torch.cat(enc_weights) if enc_weights else torch.empty(0, dtype=torch.float32),
        torch.cat(dec_indices) if dec_indices else torch.empty(0, dtype=torch.int32),
        torch.cat(dec_offsets) if dec_offsets else torch.empty(0, dtype=torch.int32),
        torch.cat(dec_weights) if dec_weights else torch.empty(0, dtype=torch.float32),
        torch.tensor(batch_lengths, dtype=torch.int32),
        torch.tensor(targets, dtype=torch.long)
    )


# ---------------------------------------------------------
# 2. TRAINING LOOP
# ---------------------------------------------------------
def train_bc(dataset_path: str, output_dir: str, epochs: int = 10,
             batch_size: int = 256, max_samples: int = None, val_split: float = 0.1):

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{ts()} Device: {device}")
    sys.stdout.flush()

    full_dataset = PokemonBCDataset(dataset_path, max_samples=max_samples)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"{ts()} Train: {train_size:,} | Val: {val_size:,}")
    sys.stdout.flush()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model = TitanTransformer(d_model=128, nhead=4, num_layers=2).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_ckpt_path = None
    total_batches = len(train_loader)

    print(f"{ts()} Batches/epoch: {total_batches} | Starting {epochs} epoch run...\n")
    sys.stdout.flush()

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        total_train_loss = 0
        running_loss = 0.0
        epoch_start = time.time()

        for i, batch in enumerate(train_loader):
            batch_start = time.time()
            ei, eo, ew, di, do_, dw, bl, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits, _ = model(ei, eo, ew, di, do_, dw, bl)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_train_loss += loss_val
            running_loss += loss_val

            if i % 50 == 0:
                elapsed = time.time() - epoch_start
                avg_so_far = running_loss / (i + 1)
                eta_secs = (elapsed / (i + 1)) * (total_batches - i - 1) if i > 0 else 0
                eta_str = fmt_time(eta_secs) if i > 0 else "--"
                print(f"{ts()} Ep {epoch+1}/{epochs} | Batch {i:4d}/{total_batches} | "
                      f"Loss: {loss_val:.4f} | AvgLoss: {avg_so_far:.4f} | "
                      f"Elapsed: {fmt_time(elapsed)} | ETA: {eta_str}")
                sys.stdout.flush()

            del ei, eo, ew, di, do_, dw, bl, targets, logits, loss

        avg_train_loss = total_train_loss / total_batches
        epoch_time = time.time() - epoch_start

        # ---- VALIDATE ----
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ei, eo, ew, di, do_, dw, bl, targets = [b.to(device) for b in batch]
                logits, _ = model(ei, eo, ew, di, do_, dw, bl)
                loss = criterion(logits, targets)
                total_val_loss += loss.item()
                del ei, eo, ew, di, do_, dw, bl, targets, logits, loss

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

        improved = avg_val_loss < best_val_loss
        ckpt_str = ""
        if improved:
            best_val_loss = avg_val_loss
            ckpt_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_ckpt_path = f"{output_dir}/{ckpt_ts}_titan_bc.pt"
            torch.save(model.state_dict(), best_ckpt_path)
            ckpt_str = f" | [CHECKPOINT SAVED -> {best_ckpt_path}]"

        print(f"\n{ts()} {'='*70}")
        print(f"{ts()} Epoch {epoch+1}/{epochs} COMPLETE | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Best Val: {best_val_loss:.4f} | Time: {fmt_time(epoch_time)}{ckpt_str}")
        print(f"{ts()} {'='*70}\n")
        sys.stdout.flush()

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"{ts()} Training complete.")
    print(f"{ts()} Best val loss: {best_val_loss:.4f}")
    print(f"{ts()} Best checkpoint: {best_ckpt_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=str,   default="bc_dataset.pt")
    parser.add_argument("--output_dir",  type=str,   default=".")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--max_samples", type=int,   default=None)
    parser.add_argument("--val_split",   type=float, default=0.1)
    args = parser.parse_args()

    train_bc(args.dataset, args.output_dir, args.epochs,
             args.batch_size, args.max_samples, args.val_split)
