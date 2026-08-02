"""
preprocess.py – Phase 1 of the Behavioral Cloning Pipeline
============================================================
Walks all JSON files in experiments/01_baseline/dataset/matches/,
parses each observation using the baseline parser, and saves batches
of SHARD_SIZE samples as .pt files to experiments/02_behavioral_cloning/shards/.

JSON format per file:
  data = { "steps": [ step, ... ] }
  step = [player0_entry, player1_entry]
  player_entry = {
      "observation": {
          "current": { ... },   # game state
          "select": {           # None during deck selection
              "type": int,
              "context": int,
              "minCount": int,
              "maxCount": int,
              "option": [ { "type": int, ... }, ... ]
          },
          ...
      },
      "action": [int, ...]   # list of chosen option indices
  }

A sample is VALID when:
  - obs.select is not None  (skip deck-selection phase)
  - action is non-empty and len(action) == 1  (single-select decisions only)
  - obs.select.maxCount == 1  (guard: confirms single-select step)
  - The chosen index is within bounds of the option list

Each saved dict has:
  encoder_indices  : LongTensor [N_enc_tokens]
  encoder_values   : FloatTensor [N_enc_tokens]
  encoder_offsets  : LongTensor [24]   (word boundaries)
  decoder_inputs   : list of (indices, values, offsets) per action option
  target           : int  (index of chosen action in option list)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# ── Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "experiments" / "01_baseline" / "agent"
MATCHES_DIR = REPO_ROOT / "experiments" / "01_baseline" / "dataset" / "matches"
SHARDS_DIR = REPO_ROOT / "experiments" / "02_behavioral_cloning" / "shards"

sys.path.insert(0, str(AGENT_DIR))

from cg.api import to_observation_class
from parser import get_encoder_input, get_decoder_input  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
SHARD_SIZE = 1000          # samples per .pt file
SKIP_SELECT_TYPES = {8}    # COUNT type – usually deck ordering / prize selection, skip

# ── Helpers ──────────────────────────────────────────────────────────────────

def sv_to_tensors(sv):
    """Convert a SparseVector to (indices LongTensor, values FloatTensor, offsets LongTensor)."""
    indices = torch.tensor(sv.index, dtype=torch.long)
    values  = torch.tensor(sv.value,  dtype=torch.float32)
    offsets = torch.tensor(sv.offset, dtype=torch.long)
    return indices, values, offsets


def parse_sample(obs_dict: dict, action: list) -> dict | None:
    """
    Given a raw observation dict and the list of chosen option indices,
    return a sample dict or None if it should be skipped.

    Valid samples have exactly 1 chosen option (single-select decisions).
    """
    # Skip deck selection phase
    select_raw = obs_dict.get("select")
    if select_raw is None:
        return None

    # Only handle single-select decisions
    if len(action) != 1:
        return None

    target_idx = action[0]
    n_options = len(select_raw.get("option", []))

    if n_options < 1 or target_idx >= n_options:
        return None

    # Skip very trivial forced moves (1 option) — optional; keep for now
    # to preserve as much signal as possible.  Un-comment to filter:
    # if n_options == 1:
    #     return None

    # Parse observation
    obs = to_observation_class(obs_dict)
    if obs.select is None or obs.current is None:
        return None

    # Skip certain select types (e.g. COUNT)
    if int(obs.select.type) in SKIP_SELECT_TYPES:
        return None

    # ── Encoder ─────────────────────────────────────────────────────────────
    enc_sv = get_encoder_input(obs)
    assert len(enc_sv.offset) == 24, "Encoder must produce exactly 24 words"
    enc_idx, enc_val, enc_off = sv_to_tensors(enc_sv)

    # ── Decoder (one entry per option, each option = a single-element group) ─
    # Each possible action is the group [[i]] for i in range(n_options).
    actions = [[i] for i in range(n_options)]
    dec_sv = get_decoder_input(obs, actions)

    # Split decoder SparseVector back into per-option tensors.
    # dec_sv.offset[k] = start token index for option k.
    # Tokens for option k go from offset[k] to offset[k+1] (or end).
    dec_offsets = dec_sv.offset          # list of int, length == n_options
    all_idx = dec_sv.index
    all_val = dec_sv.value

    decoder_inputs = []
    for k in range(n_options):
        start = dec_offsets[k]
        end   = dec_offsets[k + 1] if k + 1 < len(dec_offsets) else len(all_idx)
        opt_idx = torch.tensor(all_idx[start:end], dtype=torch.long)
        opt_val = torch.tensor(all_val[start:end], dtype=torch.float32)
        # Per-option offset tensor is always [0] (single word)
        opt_off = torch.tensor([0], dtype=torch.long)
        decoder_inputs.append((opt_idx, opt_val, opt_off))

    return {
        "encoder_indices": enc_idx,
        "encoder_values":  enc_val,
        "encoder_offsets": enc_off,
        "decoder_inputs":  decoder_inputs,
        "target":          target_idx,
    }


def process_file(json_path: Path) -> list[dict]:
    """Parse a single match JSON and return a list of valid samples."""
    samples = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to load {json_path.name}: {e}")
        return samples

    steps = data.get("steps", [])
    for step in steps:
        # Each step has 2 player entries
        if not isinstance(step, list) or len(step) < 2:
            continue
        for player_entry in step:
            try:
                obs_dict = player_entry.get("observation", {})
                action   = player_entry.get("action", [])
                sample   = parse_sample(obs_dict, action)
                if sample is not None:
                    samples.append(sample)
            except Exception:
                # Skip bad samples silently
                pass

    return samples


def save_shard(shard: list[dict], shard_idx: int, out_dir: Path):
    """Save a list of sample dicts to a numbered .pt shard file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_idx:04d}.pt"
    torch.save(shard, path)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Pokemon TCG match replays into BC tensor shards."
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Limit the number of JSON files processed (for debugging)."
    )
    parser.add_argument(
        "--shard_size", type=int, default=SHARD_SIZE,
        help=f"Samples per shard file (default: {SHARD_SIZE})."
    )
    parser.add_argument(
        "--matches_dir", type=str, default=str(MATCHES_DIR),
        help="Path to directory of match JSON files."
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(SHARDS_DIR),
        help="Output directory for shard .pt files."
    )
    args = parser.parse_args()

    matches_dir = Path(args.matches_dir)
    out_dir     = Path(args.out_dir)
    shard_size  = args.shard_size

    json_files = sorted(matches_dir.glob("*.json"))
    if args.max_files is not None:
        json_files = json_files[: args.max_files]

    print(f"Processing {len(json_files)} JSON file(s) from {matches_dir}")
    print(f"Shards will be saved to: {out_dir}  (shard_size={shard_size})")
    print()

    current_shard: list[dict] = []
    shard_idx     = 0
    total_samples = 0
    saved_shards  = 0

    for file_i, json_path in enumerate(json_files):
        print(f"  [{file_i + 1}/{len(json_files)}] {json_path.name} ...", end="", flush=True)
        file_samples = process_file(json_path)
        print(f" {len(file_samples)} samples")

        current_shard.extend(file_samples)
        total_samples += len(file_samples)

        # Flush full shards
        while len(current_shard) >= shard_size:
            path = save_shard(current_shard[:shard_size], shard_idx, out_dir)
            print(f"    => Saved {path.name}")
            current_shard = current_shard[shard_size:]
            shard_idx += 1
            saved_shards += 1

    # Flush remainder
    if current_shard:
        path = save_shard(current_shard, shard_idx, out_dir)
        print(f"    => Saved {path.name} ({len(current_shard)} samples, partial)")
        saved_shards += 1

    print()
    print("=" * 60)
    print(f"Total samples extracted : {total_samples}")
    print(f"Shards saved            : {saved_shards}")
    print(f"Output directory        : {out_dir}")

    # Quick sanity check on first shard
    if saved_shards > 0:
        first_shard_path = out_dir / "shard_0000.pt"
        if first_shard_path.exists():
            shard_data = torch.load(first_shard_path, weights_only=False)
            s0 = shard_data[0]
            print()
            print("=== Sample [0] from shard_0000.pt ===")
            print(f"  encoder_indices  shape : {s0['encoder_indices'].shape}")
            print(f"  encoder_values   shape : {s0['encoder_values'].shape}")
            print(f"  encoder_offsets  shape : {s0['encoder_offsets'].shape}")
            print(f"  decoder_inputs   len   : {len(s0['decoder_inputs'])} options")
            print(f"  target                 : {s0['target']}")
            d0 = s0["decoder_inputs"][0]
            print(f"  decoder_inputs[0]      : idx={d0[0].shape}, val={d0[1].shape}, off={d0[2].shape}")


if __name__ == "__main__":
    main()
