# -*- coding: utf-8 -*-
"""
Overfit Sanity Check - TitanTransformer
========================================
Purpose: Verify that the model can memorize a SINGLE synthetic batch.
Success criteria:
    - Loss approaches 0.0
    - Accuracy reaches 1.0 (100%)

No real data needed. Synthetic data matches the REAL shard / collate format:
    enc_indices  : LongTensor  [B, max_enc_tokens]  -- zero-padded
    enc_values   : FloatTensor [B, max_enc_tokens]  -- zero-padded
    enc_offsets  : LongTensor  [B, 24]              -- per-sample bag starts (relative, 0-indexed)
    decoder_inputs : list[list[tuple(LongTensor, FloatTensor)]]
                     outer: B, inner: n_real_actions
    action_mask  : BoolTensor  [B, max_N]           -- True = invalid
    target       : LongTensor  [B]                  -- hardcoded to 2

Run:
    python experiments/02_behavioral_cloning/test_overfit.py
"""

import sys
import os
import random

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_baseline_agent = os.path.abspath(os.path.join(_this_dir, "../01_baseline/agent"))
if _baseline_agent not in sys.path:
    sys.path.insert(0, _baseline_agent)

from parser import encoder_size, decoder_size  # noqa: E402
from model import TitanTransformer, TitanConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic batch construction
# ---------------------------------------------------------------------------

def make_synthetic_batch(
    batch_size: int = 4,
    n_actions: int = 5,
    target_action: int = 2,
    seed: int = 42,
) -> dict:
    """
    Builds a fixed synthetic batch that exactly matches the real shard format.

    Encoder format (matching real collate output):
        enc_indices  : [B, max_enc_tokens]  -- zero-padded 2D tensor
        enc_values   : [B, max_enc_tokens]  -- zero-padded 2D tensor
        enc_offsets  : [B, 24]              -- per-sample word boundaries
                       Each row lists the 24 bag-start positions RELATIVE
                       to that sample's own token row (i.e. offset[b, 0] == 0
                       always, and max value < max_enc_tokens).

    Decoder format:
        decoder_inputs : list[list[tuple(LongTensor, FloatTensor)]]
                         Outer: B samples. Inner: n_actions tuples.
                         Each tuple = (1D idx tensor, 1D val tensor) for one action.

    Other:
        action_mask  : [B, max_N]  Bool -- True = padded/invalid
        target       : [B]         Long -- all equal to target_action
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Use small vocab slices for well-conditioned gradients
    enc_vocab = max(encoder_size // 20, 50)
    dec_vocab = max(decoder_size // 20, 50)

    N_WORDS = 24
    max_actions = 64  # matches TitanConfig.max_actions

    # ---- Build per-sample encoder data ---------------------------------
    # We give each sample a variable number of tokens per word (1-3),
    # then pad all samples to the same max_enc_tokens.

    enc_indices_list = []   # per sample: list of all token indices
    enc_values_list  = []   # per sample: list of all token weights
    enc_offsets_list = []   # per sample: list of 24 bag-start positions

    for _b in range(batch_size):
        sample_idx = []
        sample_val = []
        sample_off = []
        cursor = 0
        for _w in range(N_WORDS):
            sample_off.append(cursor)
            n_tok = rng.randint(1, 3)
            for _ in range(n_tok):
                sample_idx.append(rng.randint(0, enc_vocab - 1))
                sample_val.append(round(rng.uniform(0.1, 1.0), 3))
            cursor += n_tok
        enc_indices_list.append(sample_idx)
        enc_values_list.append(sample_val)
        enc_offsets_list.append(sample_off)

    # Pad all samples to max_enc_tokens with zeros
    max_enc_tokens = max(len(s) for s in enc_indices_list)
    enc_indices_2d = torch.zeros(batch_size, max_enc_tokens, dtype=torch.long)
    enc_values_2d  = torch.zeros(batch_size, max_enc_tokens, dtype=torch.float32)
    for b in range(batch_size):
        n = len(enc_indices_list[b])
        enc_indices_2d[b, :n] = torch.tensor(enc_indices_list[b], dtype=torch.long)
        enc_values_2d[b,  :n] = torch.tensor(enc_values_list[b],  dtype=torch.float32)

    enc_offsets_2d = torch.tensor(enc_offsets_list, dtype=torch.long)  # [B, 24]

    # ---- Build decoder_inputs ------------------------------------------
    # Each sample gets n_actions real action bags.
    # Each action bag has 1-2 tokens.
    
    cursor = 0
    dec_offsets_list = []
    all_dec_idx = []
    all_dec_val = []
    for _b in range(batch_size):
        for _a in range(max_actions):
            dec_offsets_list.append(cursor)
            if _a < n_actions:
                n_tok = rng.randint(1, 2)
                idx = torch.tensor(
                    [rng.randint(0, dec_vocab - 1) for _ in range(n_tok)],
                    dtype=torch.long
                )
                val = torch.ones(n_tok, dtype=torch.float32)
                all_dec_idx.append(idx)
                all_dec_val.append(val)
                cursor += n_tok

    dec_indices = torch.cat(all_dec_idx) if all_dec_idx else torch.zeros(1, dtype=torch.long)
    dec_values = torch.cat(all_dec_val) if all_dec_val else torch.zeros(1, dtype=torch.float32)
    dec_offsets_2d = torch.tensor(dec_offsets_list, dtype=torch.long)

    # ---- Action mask (True = invalid/padded) ---------------------------
    action_mask = torch.ones(batch_size, max_actions, dtype=torch.bool)
    action_mask[:, :n_actions] = False  # first n_actions slots are valid

    # ---- Target --------------------------------------------------------
    target = torch.full((batch_size,), target_action, dtype=torch.long)

    return {
        "enc_indices":    enc_indices_2d,   # [B, max_enc_tokens]
        "enc_values":     enc_values_2d,    # [B, max_enc_tokens]
        "enc_offsets":    enc_offsets_2d,   # [B, 24]
        "dec_indices":    dec_indices,
        "dec_values":     dec_values,
        "dec_offsets":    dec_offsets_2d,
        "action_mask":    action_mask,      # [B, max_N]
        "target":         target,           # [B]
    }


# ---------------------------------------------------------------------------
# Overfit loop
# ---------------------------------------------------------------------------

def run_overfit_test(
    n_epochs: int = 200,
    lr: float = 1e-2,
    batch_size: int = 4,
    n_actions: int = 5,
    target_action: int = 2,
):
    print("=" * 65)
    print("  TitanTransformer - Overfit Sanity Check")
    print("=" * 65)

    # ---- Model ---------------------------------------------------------
    cfg = TitanConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.0,   # disable dropout for overfit test
        max_actions=64,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TitanTransformer(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  encoder_size  : {encoder_size:,}")
    print(f"  decoder_size  : {decoder_size:,}")
    print(f"  Parameters    : {n_params:,}")
    print(f"  Device        : {device}")
    print(f"  Batch size    : {batch_size}")
    print(f"  n_actions     : {n_actions}")
    print(f"  Target action : {target_action}")
    print(f"  Epochs        : {n_epochs}")
    print(f"  LR            : {lr}")
    print()

    # ---- Synthetic data -----------------------------------------------
    batch = make_synthetic_batch(
        batch_size=batch_size,
        n_actions=n_actions,
        target_action=target_action,
    )

    # Move tensor fields to device
    enc_indices = batch["enc_indices"].to(device)
    enc_values  = batch["enc_values"].to(device)
    enc_offsets = batch["enc_offsets"].to(device)
    dec_indices = batch["dec_indices"].to(device)
    dec_values  = batch["dec_values"].to(device)
    dec_offsets = batch["dec_offsets"].to(device)
    action_mask = batch["action_mask"].to(device)
    target      = batch["target"].to(device)

    # ---- Optimizer & loss ---------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"  {'Epoch':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")

    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        logits = model(
            enc_indices,
            enc_values,
            enc_offsets,
            dec_indices,
            dec_values,
            dec_offsets,
            action_mask,
        )  # [B, max_actions]

        # Only score the valid action slots (first n_actions)
        loss = criterion(logits[:, :n_actions], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits[:, :n_actions].argmax(dim=-1)
        acc = (preds == target).float().mean().item()

        loss_val = loss.item()
        history.append((epoch, loss_val, acc))

        print(f"  {epoch:>6}  {loss_val:>10.6f}  {acc:>10.4f}")

    # ---- Summary -------------------------------------------------------
    print()
    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    final_loss, final_acc = history[-1][1], history[-1][2]

    loss_ok = final_loss < 0.01
    acc_ok  = final_acc  >= 1.0

    loss_status = "PASS (< 0.01)" if loss_ok else "FAIL (>= 0.01)"
    acc_status  = "PASS (= 1.0)"  if acc_ok  else "FAIL (< 1.0)"
    print(f"\n  Final loss     : {final_loss:.6f}  [{loss_status}]")
    print(f"  Final accuracy : {final_acc:.4f}    [{acc_status}]")

    overall = loss_ok and acc_ok
    print()
    overall_str = "OVERFIT TEST PASSED" if overall else "OVERFIT TEST FAILED"
    print(f"  OVERALL: {overall_str}")
    print("=" * 65)

    return history, n_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    history, n_params = run_overfit_test(
        n_epochs=200,
        lr=1e-2,
        batch_size=4,
        n_actions=5,
        target_action=2,
    )
