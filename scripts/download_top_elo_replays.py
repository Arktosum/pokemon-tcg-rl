"""
download_top_elo_replays.py
===========================
Downloads every available public replay from the TOP 10 leaderboard
teams (1130+ Elo) in the pokemon-tcg-ai-battle competition.

Outputs:
    data/replays/              - One JSON per episode
    data/replay_metadata.csv   - Full metadata table (all games)
    data/bc_dataset.jsonl      - (state_vec, action) pairs for BC training
"""

import subprocess
import json
import csv
import os
import sys
import time
from pathlib import Path

COMPETITION    = "pokemon-tcg-ai-battle"
MIN_SUB_SCORE  = 1130.0  # Only download replays from submissions at or above this Elo
REPO_ROOT   = Path(__file__).resolve().parent.parent
REPLAY_DIR  = REPO_ROOT / "data" / "replays"
CSV_PATH    = REPO_ROOT / "data" / "replay_metadata.csv"

# ── Top-10 leaderboard teams (verified from Phase 46 raw output) ─────────────
TOP_ELO_TEAMS = [
    {"team_id": "16514272", "team_name": "Dominic Peel",                    "elo": 1153.4},
    {"team_id": "16380946", "team_name": "flg",                             "elo": 1152.8},
    {"team_id": "16463316", "team_name": "LiamK",                           "elo": 1152.1},
    {"team_id": "16408505", "team_name": "JZ",                              "elo": 1150.4},
    {"team_id": "16391123", "team_name": "Iliamna",                         "elo": 1147.0},
    {"team_id": "16531269", "team_name": "Dries @ Tufa Labs",               "elo": 1141.2},
    {"team_id": "16381823", "team_name": "Yushin Ito",                      "elo": 1139.8},
    {"team_id": "16375647", "team_name": "James Cox",                       "elo": 1139.3},
    {"team_id": "16386872", "team_name": "titako0000",                      "elo": 1135.0},
    {"team_id": "16464757", "team_name": "wwwwwwwwwwwwwwwwwwwwwwwwwwwwww", "elo": 1134.3},
]

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.stdout.strip(), result.stderr.strip()

def get_team_submissions(team_id):
    """Get all public submissions for a team."""
    cmd = f"kaggle competitions team-submissions {team_id} --format csv"
    stdout, stderr = run(cmd)
    subs = []
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    header_found = False
    for line in lines:
        if line.startswith("id,"):
            header_found = True
            continue
        if not header_found:
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            subs.append({
                "submission_id": parts[0].strip(),
                "date":          parts[1].strip(),
                "score":         parts[2].strip() if len(parts) > 2 else "",
            })
    return subs

def list_episodes(submission_id):
    """Get all episodes for a submission."""
    cmd = f"kaggle competitions episodes {submission_id} --format csv"
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
                "episode_id":     parts[0].strip(),
                "create_time":    parts[1].strip(),
                "end_time":       parts[2].strip(),
                "state":          parts[3].strip(),
                "episode_type":   parts[4].strip(),
            })
    return episodes

def download_replay(episode_id, out_dir, retries=3):
    """Download replay JSON. Kaggle saves as: episode-{id}-replay.json"""
    expected = out_dir / f"episode-{episode_id}-replay.json"
    if expected.exists() and expected.stat().st_size > 100:
        return expected, "cached"
    for attempt in range(retries):
        cmd = f"kaggle competitions replay {episode_id} -p {out_dir} -q"
        _, stderr = run(cmd)
        if expected.exists() and expected.stat().st_size > 100:
            return expected, "downloaded"
        if "429" in stderr or "Too Many Requests" in stderr:
            wait = 8 * (attempt + 1)
            print(f"    [429] Rate limited. Waiting {wait}s...", flush=True)
            time.sleep(wait)
        else:
            # Non-rate-limit failure, short wait and retry
            time.sleep(2)
    return None, f"failed after {retries} attempts"

def extract_meta(replay_path, ep_meta, sub_meta, team_meta):
    """Extract structured metadata from a replay JSON."""
    row = {
        "episode_id":       ep_meta["episode_id"],
        "team_id":          team_meta["team_id"],
        "team_name":        team_meta["team_name"],
        "team_elo":         team_meta["elo"],
        "submission_id":    sub_meta["submission_id"],
        "submission_date":  sub_meta["date"],
        "submission_score": sub_meta["score"],
        "create_time":      ep_meta["create_time"],
        "end_time":         ep_meta["end_time"],
        "episode_state":    ep_meta["state"],
        "episode_type":     ep_meta["episode_type"],
        "duration_seconds": "",
        "player0_name":     "",
        "player0_score":    "",
        "player0_status":   "",
        "player1_name":     "",
        "player1_score":    "",
        "player1_status":   "",
        "winner":           "",
        "total_steps":      "",
        "replay_file":      str(replay_path) if replay_path else "",
        "error":            "",
    }
    if not replay_path:
        return row
    try:
        with open(replay_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        steps = data.get("steps", [])
        row["total_steps"] = len(steps)

        agents = data.get("agents", [])
        if len(agents) >= 1:
            row["player0_name"]   = agents[0].get("name", "")
            row["player0_score"]  = agents[0].get("reward", "")
            row["player0_status"] = agents[0].get("status", "")
        if len(agents) >= 2:
            row["player1_name"]   = agents[1].get("name", "")
            row["player1_score"]  = agents[1].get("reward", "")
            row["player1_status"] = agents[1].get("status", "")

        # Determine winner
        if len(agents) >= 2:
            try:
                r0, r1 = float(agents[0].get("reward", 0) or 0), float(agents[1].get("reward", 0) or 0)
                if r0 > r1:
                    row["winner"] = "player0"
                elif r1 > r0:
                    row["winner"] = "player1"
                else:
                    row["winner"] = "draw"
            except (TypeError, ValueError):
                pass

        # Duration
        try:
            from datetime import datetime
            fmt = "%Y-%m-%d %H:%M:%S"
            ct = ep_meta["create_time"].split(".")[0]
            et = ep_meta["end_time"].split(".")[0]
            row["duration_seconds"] = int(
                (datetime.strptime(et, fmt) - datetime.strptime(ct, fmt)).total_seconds()
            )
        except Exception:
            pass

    except Exception as e:
        row["error"] = str(e)[:200]
    return row

def main():
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    # ── PHASE 1: Enumerate all episodes ──────────────────────────────────────
    print("=" * 65)
    print("PHASE 1: Enumerating episodes from Top-10 Elo teams")
    print("=" * 65, flush=True)

    all_work = []  # list of (ep_meta, sub_meta, team_meta)

    for team in TOP_ELO_TEAMS:
        print(f"\n  [{team['elo']}] {team['team_name']} (ID: {team['team_id']})", flush=True)
        subs = get_team_submissions(team["team_id"])
        print(f"    Submissions found: {len(subs)}", flush=True)
        team_episode_count = 0
        subs_clean = []
        for sub in subs:
            try:
                score = float(sub["score"])
            except (ValueError, TypeError):
                score = 0.0
            if score >= MIN_SUB_SCORE:
                subs_clean.append(sub)
            else:
                print(f"      SKIP Sub {sub['submission_id']} (score={sub['score']} < {MIN_SUB_SCORE})", flush=True)
        subs = subs_clean
        for sub in subs:
            time.sleep(0.4)
            episodes = list_episodes(sub["submission_id"])
            print(f"      Sub {sub['submission_id']} (score={sub['score']}): {len(episodes)} episodes", flush=True)
            team_episode_count += len(episodes)
            for ep in episodes:
                all_work.append((ep, sub, team))
            time.sleep(0.4)
        print(f"    Total episodes this team: {team_episode_count}", flush=True)

    total = len(all_work)
    print(f"\nTOTAL EPISODES TO DOWNLOAD: {total}", flush=True)

    # ── PHASE 2: Download replays ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 2: Downloading replays (sequential, rate-limit aware)")
    print("=" * 65, flush=True)

    downloaded, cached, failed = 0, 0, 0
    all_meta_rows = []

    for i, (ep_meta, sub_meta, team_meta) in enumerate(all_work, 1):
        ep_id = ep_meta["episode_id"]
        path, status = download_replay(ep_id, REPLAY_DIR)

        if status == "cached":
            cached += 1
        elif status == "downloaded":
            downloaded += 1
            time.sleep(1.0)   # polite delay after actual download
        else:
            failed += 1
            time.sleep(0.5)

        row = extract_meta(path, ep_meta, sub_meta, team_meta)
        if not path:
            row["error"] = status
        all_meta_rows.append(row)

        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} | downloaded={downloaded} cached={cached} failed={failed}", flush=True)

    # ── PHASE 3: Write CSV ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3: Writing replay_metadata.csv")
    print("=" * 65, flush=True)

    fieldnames = [
        "episode_id", "team_id", "team_name", "team_elo",
        "submission_id", "submission_date", "submission_score",
        "create_time", "end_time", "duration_seconds",
        "episode_state", "episode_type",
        "player0_name", "player0_score", "player0_status",
        "player1_name", "player1_score", "player1_status",
        "winner", "total_steps",
        "replay_file", "error",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_meta_rows)

    print(f"\n  Written: {len(all_meta_rows)} rows -> {CSV_PATH}", flush=True)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    wins_p0 = sum(1 for r in all_meta_rows if r["winner"] == "player0")
    wins_p1 = sum(1 for r in all_meta_rows if r["winner"] == "player1")
    draws    = sum(1 for r in all_meta_rows if r["winner"] == "draw")

    print(f"""
SUMMARY
  Teams scraped:      {len(TOP_ELO_TEAMS)} (Top 10, 1130+ Elo)
  Total episodes:     {total}
  Downloaded:         {downloaded}
  Cached:             {cached}
  Failed:             {failed}
  Player0 wins:       {wins_p0}
  Player1 wins:       {wins_p1}
  Draws:              {draws}
  CSV:                {CSV_PATH}
  Replay dir:         {REPLAY_DIR}
  BC model target:    TOP_ELO_BC_MODEL
""", flush=True)

if __name__ == "__main__":
    main()
