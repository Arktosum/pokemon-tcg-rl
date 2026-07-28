"""
download_all_replays.py
=======================
Bulk-download every available replay from every COMPLETE submission
for the pokemon-tcg-ai-battle competition, then write a full metadata CSV.

Kaggle CLI saves replays as: episode-{id}-replay.json
"""

import subprocess
import json
import csv
import os
import sys
import time
from pathlib import Path

COMPETITION = "pokemon-tcg-ai-battle"
REPO_ROOT   = Path(__file__).resolve().parent.parent
REPLAY_DIR  = REPO_ROOT / "data" / "replays"
CSV_PATH    = REPO_ROOT / "data" / "replay_metadata.csv"

# All submission IDs with COMPLETE status
COMPLETE_SUBMISSIONS = [
    {"ref": "55060308", "desc": "Phase 44 True TITAN"},
    {"ref": "55060096", "desc": "Phase 42 True TITAN"},
    {"ref": "55059474", "desc": "Phase 42 True TITAN (2)"},
    {"ref": "55059382", "desc": "Phase 41 Vanilla"},
    {"ref": "55030967", "desc": "Alakazam 5th Place"},
    {"ref": "55029836", "desc": "Rule-Based Archaludon Bot"},
    {"ref": "55022405", "desc": "Fixed C++ Engine Import"},
    {"ref": "55008620", "desc": "V5.1 64pct WR Ep4825"},
    {"ref": "55007915", "desc": "V5.1 64pct WR Ep3740"},
    {"ref": "55007191", "desc": "V5.1 60pct WR Ep2385"},
    {"ref": "55006785", "desc": "V5.1 Baseline 48pct"},
    {"ref": "55000925", "desc": "PPO Transformer v1"},
    {"ref": "54985186", "desc": "LSTM ActorCritic 500 Epochs"},
    {"ref": "54982376", "desc": "RL Recurrent ActorCritic 500 Epochs"},
]

def run(cmd):
    """Run shell command, return (stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.stdout.strip(), result.stderr.strip()

def list_episodes(sub_ref):
    """Return list of episode dicts for a submission."""
    cmd = f"kaggle competitions episodes {sub_ref} --format csv"
    stdout, stderr = run(cmd)
    episodes = []
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    header_found = False
    for line in lines:
        if line.startswith("id,"):
            header_found = True
            continue
        if not header_found:
            continue
        if line.startswith("Use "):
            break
        parts = line.split(",")
        if len(parts) >= 5:
            episodes.append({
                "episode_id":  parts[0].strip(),
                "create_time": parts[1].strip(),
                "end_time":    parts[2].strip(),
                "state":       parts[3].strip(),
                "type":        parts[4].strip(),
            })
    return episodes

def download_replay(episode_id, out_dir):
    """
    Download replay. Kaggle saves as: episode-{id}-replay.json
    Returns (path, status_str).
    """
    expected_path = out_dir / f"episode-{episode_id}-replay.json"
    if expected_path.exists() and expected_path.stat().st_size > 0:
        return expected_path, "cached"

    cmd = f"kaggle competitions replay {episode_id} -p {out_dir} -q"
    stdout, stderr = run(cmd)

    if expected_path.exists() and expected_path.stat().st_size > 0:
        return expected_path, "downloaded"

    # Rate limit: back off and retry once
    if "429" in stderr or "Too Many Requests" in stderr:
        time.sleep(5)
        run(cmd)
        if expected_path.exists() and expected_path.stat().st_size > 0:
            return expected_path, "downloaded_retry"
        return None, f"rate_limited: {stderr[:100]}"

    return None, f"failed: {stderr[:100]}"

def extract_replay_meta(replay_path, episode_meta, sub_meta):
    """Parse replay JSON and extract structured metadata."""
    row = {
        "episode_id":      episode_meta["episode_id"],
        "submission_ref":  sub_meta["ref"],
        "submission_desc": sub_meta["desc"],
        "create_time":     episode_meta["create_time"],
        "end_time":        episode_meta["end_time"],
        "state":           episode_meta["state"],
        "type":            episode_meta["type"],
        "replay_file":     str(replay_path),
        "player0_name":    "",
        "player0_score":   "",
        "player0_status":  "",
        "player1_name":    "",
        "player1_score":   "",
        "player1_status":  "",
        "winner":          "",
        "turn_count":      "",
        "total_steps":     "",
        "duration_seconds":"",
        "error":           "",
    }
    try:
        with open(replay_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        steps = data.get("steps", [])
        row["total_steps"] = len(steps)
        row["turn_count"]  = len(steps)

        agents = data.get("agents", [])
        if len(agents) >= 1:
            row["player0_name"]   = agents[0].get("name", "")
            row["player0_score"]  = agents[0].get("reward", "")
            row["player0_status"] = agents[0].get("status", "")
        if len(agents) >= 2:
            row["player1_name"]   = agents[1].get("name", "")
            row["player1_score"]  = agents[1].get("reward", "")
            row["player1_status"] = agents[1].get("status", "")

        # Winner
        if len(agents) >= 2:
            r0 = agents[0].get("reward")
            r1 = agents[1].get("reward")
            if r0 is not None and r1 is not None:
                try:
                    if float(r0) > float(r1):
                        row["winner"] = f"player0"
                    elif float(r1) > float(r0):
                        row["winner"] = f"player1"
                    else:
                        row["winner"] = "draw"
                except (TypeError, ValueError):
                    pass

        # Duration
        try:
            from datetime import datetime
            ct = episode_meta["create_time"].split(".")[0]
            et = episode_meta["end_time"].split(".")[0]
            fmt = "%Y-%m-%d %H:%M:%S"
            dur = (datetime.strptime(et, fmt) - datetime.strptime(ct, fmt)).total_seconds()
            row["duration_seconds"] = int(dur)
        except Exception:
            pass

    except Exception as e:
        row["error"] = str(e)[:200]
    return row

def main():
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata_rows = []
    all_episodes = []

    print("=" * 60)
    print("PHASE 1: Enumerating all episodes")
    print("=" * 60)
    for sub in COMPLETE_SUBMISSIONS:
        print(f"  Sub {sub['ref']} - {sub['desc']}", flush=True)
        episodes = list_episodes(sub["ref"])
        print(f"    => {len(episodes)} episodes", flush=True)
        for ep in episodes:
            all_episodes.append((ep, sub))
        time.sleep(0.5)

    total = len(all_episodes)
    print(f"\nTotal episodes: {total}", flush=True)

    print("\n" + "=" * 60)
    print("PHASE 2: Downloading replays (sequential, 1s delay)")
    print("=" * 60)

    downloaded = 0
    failed = 0
    cached = 0

    for i, (ep_meta, sub_meta) in enumerate(all_episodes, 1):
        ep_id = ep_meta["episode_id"]
        path, status = download_replay(ep_id, REPLAY_DIR)

        if status == "cached":
            cached += 1
        elif "downloaded" in status:
            downloaded += 1
            time.sleep(1.2)  # polite delay only after actual download
        else:
            failed += 1
            time.sleep(0.5)

        if i % 20 == 0 or i == total:
            print(f"  {i}/{total} | dl={downloaded} cached={cached} failed={failed}", flush=True)

        if path:
            row = extract_replay_meta(path, ep_meta, sub_meta)
        else:
            row = {
                "episode_id":       ep_meta["episode_id"],
                "submission_ref":   sub_meta["ref"],
                "submission_desc":  sub_meta["desc"],
                "create_time":      ep_meta["create_time"],
                "end_time":         ep_meta["end_time"],
                "state":            ep_meta["state"],
                "type":             ep_meta["type"],
                "replay_file":      "",
                "player0_name":     "",
                "player0_score":    "",
                "player0_status":   "",
                "player1_name":     "",
                "player1_score":    "",
                "player1_status":   "",
                "winner":           "",
                "turn_count":       "",
                "total_steps":      "",
                "duration_seconds": "",
                "error":            status,
            }
        all_metadata_rows.append(row)

    print("\n" + "=" * 60)
    print("PHASE 3: Writing CSV")
    print("=" * 60)

    fieldnames = [
        "episode_id", "submission_ref", "submission_desc",
        "create_time", "end_time", "duration_seconds",
        "state", "type",
        "player0_name", "player0_score", "player0_status",
        "player1_name", "player1_score", "player1_status",
        "winner", "turn_count", "total_steps",
        "replay_file", "error",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata_rows)

    print(f"\n  Wrote {len(all_metadata_rows)} rows => {CSV_PATH}", flush=True)
    print(f"\nSUMMARY")
    print(f"  Total episodes:  {total}")
    print(f"  Downloaded:      {downloaded}")
    print(f"  Cached:          {cached}")
    print(f"  Failed:          {failed}")
    print(f"  CSV:             {CSV_PATH}")

if __name__ == "__main__":
    main()
