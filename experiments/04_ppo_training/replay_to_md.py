#!/usr/bin/env python3
"""
replay_to_md.py

Parses a "cabt" (Card Battle) Kaggle-style TCG replay JSON file and writes a
human-readable, step-by-step Markdown breakdown of the match.

Event decoding is written against the authoritative cabt Engine API docs:
https://matsuoinstitute.github.io/cabt/api.html (Enums: LogType, AreaType).
Every documented LogType member is decoded explicitly in decode_entry_logs();
the fallback branch there only fires for a `type` string outside that enum
(e.g. a future engine version adding a new log type).
"""

import sys
import json
import collections
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

@dataclass
class ReplayParseInput:
    raw_json: Dict[str, Any]

@dataclass
class ReplayParseOutput:
    markdown: str

def build_id_name_map(obj: Any) -> Dict[int, str]:
    """Walk the whole JSON tree and collect {card id: card name} pairs."""
    id_name: Dict[int, str] = {}

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if isinstance(o.get("id"), int) and isinstance(o.get("name"), str):
                id_name[o["id"]] = o["name"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return id_name

def find_setup_visualize(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = data.get("steps", [])
    for step in steps:
        for agent in step:
            viz = agent.get("visualize")
            if viz and isinstance(viz, list) and len(viz) > 0:
                if all(isinstance(e, dict) and "logs" in e for e in viz):
                    return viz
    return []

def card_label(id_name: Dict[int, str], card_id: Optional[int], serial: Optional[int] = None) -> str:
    name = id_name.get(card_id, f"Card#{card_id}") if card_id is not None else "Unknown Card"
    if serial is not None:
        return f"{name} (#{serial})"
    return name

def player_label(player_index: Optional[int]) -> str:
    if player_index is None:
        return "A player"
    return f"Player {player_index + 1}"

# Source: https://matsuoinstitute.github.io/cabt/api.html#api.AreaType
# Zones used in MoveCard/MoveCardReverse fromArea/toArea fields.
AREA_NAMES: Dict[int, str] = {
    1: "Deck",
    2: "Hand",
    3: "Discard",
    4: "Active",
    5: "Bench",
    6: "Prize",
    7: "Stadium",
    8: "Energy",
    9: "Tool",
    10: "Pre-Evolution",
    11: "Player",
    12: "Looking",
}

def area_name(area: Optional[int]) -> str:
    if area is None:
        return "Unknown Zone"
    return AREA_NAMES.get(area, f"Zone {area} (undocumented AreaType)")

def decode_entry_logs(entry_logs: List[Dict[str, Any]], id_name: Dict[int, str]) -> List[str]:
    """
    Decodes raw Log entries per the documented LogType enum:
    https://matsuoinstitute.github.io/cabt/api.html#api.LogType

    Every LogType member (24 total) is handled explicitly below. The fallback
    branch only fires for a type string the documented enum does not define
    (e.g. if the engine is updated with a new log type in the future).
    """
    lines: List[str] = []
    i = 0
    n = len(entry_logs)

    def cl(e: Dict[str, Any], key_id: str = "cardId", key_serial: str = "serial") -> str:
        return card_label(id_name, e.get(key_id), e.get(key_serial))

    def status_line(e: Dict[str, Any], pi, condition_name: str) -> str:
        # Shared decoder for POISONED/BURNED/ASLEEP/PARALYZED/CONFUSED, which all
        # share the same {playerIndex, isRecover, cardId, serial} field shape.
        target = cl(e)
        if e.get("isRecover"):
            return f"{player_label(pi)}'s {target} recovers from {condition_name}."
        return f"{player_label(pi)}'s {target} becomes {condition_name}."

    while i < n:
        e = entry_logs[i]
        t = e.get("type")
        pi = e.get("playerIndex")

        # --- SHUFFLE (0) ---
        if t == "Shuffle":
            lines.append(f"{player_label(pi)} shuffles their deck.")

        # --- HAS_BASIC_POKEMON (1) ---
        elif t == "HasBasicPokemon":
            status = "has" if e.get("hasBasicPokemon") else "does NOT have"
            lines.append(f"{player_label(pi)} {status} a Basic Pokémon in their opening hand.")

        # --- TURN_START (2) ---
        elif t == "TurnStart":
            lines.append(f"###TURNDIV###-- {player_label(pi)}'s turn begins --")

        # --- TURN_END (3) ---
        elif t == "TurnEnd":
            lines.append(f"###TURNDIV###-- {player_label(pi)}'s turn ends --")

        # --- DRAW (4) ---
        elif t == "Draw":
            if "cardId" in e:
                lines.append(f"{player_label(pi)} draws {cl(e)}.")
            else:
                lines.append(f"{player_label(pi)} draws a card.")

        # --- DRAW_REVERSE (5): opponent drew a card, identity hidden from us ---
        elif t == "DrawReverse":
            lines.append(f"{player_label(pi)} draws a card from their deck (face-down).")

        # --- MOVE_CARD (6) ---
        elif t == "MoveCard":
            fr, to = area_name(e.get("fromArea")), area_name(e.get("toArea"))
            card_txt = cl(e) if "cardId" in e else "a card"
            lines.append(f"{player_label(pi)} moves {card_txt} ({fr} -> {to}).")

        # --- MOVE_CARD_REVERSE (7): a face-down card moved, identity hidden ---
        elif t == "MoveCardReverse":
            fr, to = area_name(e.get("fromArea")), area_name(e.get("toArea"))
            lines.append(f"{player_label(pi)} moves a face-down card ({fr} -> {to}).")

        # --- SWITCH (8) ---
        elif t == "Switch":
            # Per docs: cardIdActive/serialActive = Pokémon moving TO Bench (i.e.
            # the outgoing Active); cardIdBench/serialBench = Pokémon moving TO
            # Active (i.e. the incoming one, previously benched).
            outgoing = cl(e, "cardIdActive", "serialActive")
            incoming = cl(e, "cardIdBench", "serialBench")
            lines.append(
                f"{player_label(pi)} switches their Active Pokémon: "
                f"{incoming} becomes Active (was on Bench); {outgoing} moves to the Bench."
            )

        # --- CHANGE (9) ---
        elif t == "Change":
            before = cl(e, "cardIdBefore", "serialBefore")
            after = cl(e, "cardIdAfter", "serialAfter")
            lines.append(f"{player_label(pi)}'s {before} changes into {after}.")

        # --- PLAY (10) ---
        elif t == "Play":
            lines.append(f"{player_label(pi)} plays {cl(e)}.")

        # --- ATTACH (11) ---
        elif t == "Attach":
            lines.append(f"{player_label(pi)} attaches {cl(e)} to {cl(e, 'cardIdTarget', 'serialTarget')}.")

        # --- EVOLVE (12) ---
        elif t == "Evolve":
            lines.append(f"{player_label(pi)} evolves {cl(e, 'cardIdTarget', 'serialTarget')} into {cl(e)}.")

        # --- DEVOLVE (13) ---
        elif t == "Devolve":
            lines.append(f"{player_label(pi)} devolves {cl(e, 'cardIdTarget', 'serialTarget')} into {cl(e)}.")

        # --- MOVE_ATTACHED (14) ---
        elif t == "MoveAttached":
            attached = cl(e)
            before = cl(e, "cardIdBefore", "serialBefore")
            after = cl(e, "cardIdAfter", "serialAfter")
            lines.append(f"{player_label(pi)} moves attached {attached} from {before} to {after}.")

        # --- ATTACK (15); consumes any immediately-following HP_CHANGE entries
        #     from the same logs list as the attack's damage/effects ---
        elif t == "Attack":
            attacker = cl(e)
            attack_id = e.get("attackId")
            dmg_lines: List[str] = []
            j = i + 1
            while j < n and entry_logs[j].get("type") == "HpChange":
                he = entry_logs[j]
                target = cl(he)
                val = he.get("value")
                verb = "loses" if (val or 0) < 0 else "gains"
                counter_note = " (damage counter placed)" if he.get("putDamageCounter") else ""
                dmg_lines.append(
                    f"{player_label(he.get('playerIndex'))}'s {target} {verb} "
                    f"{abs(val) if val is not None else '?'} HP{counter_note}."
                )
                j += 1
            lines.append(f"{player_label(pi)}'s {attacker} attacks (attack ID {attack_id}).")
            lines.extend(f"  - {d}" for d in dmg_lines)
            i = j - 1

        # --- HP_CHANGE (16), when not consumed as part of a preceding Attack ---
        elif t == "HpChange":
            target = cl(e)
            val = e.get("value")
            verb = "loses" if (val or 0) < 0 else "gains"
            counter_note = " (damage counter placed)" if e.get("putDamageCounter") else ""
            lines.append(f"{player_label(pi)}'s {target} {verb} {abs(val) if val is not None else '?'} HP{counter_note}.")

        # --- POISONED / BURNED / ASLEEP / PARALYZED / CONFUSED (17-21) ---
        elif t == "Poisoned":
            lines.append(status_line(e, pi, "Poisoned"))
        elif t == "Burned":
            lines.append(status_line(e, pi, "Burned"))
        elif t == "Asleep":
            lines.append(status_line(e, pi, "Asleep"))
        elif t == "Paralyzed":
            lines.append(status_line(e, pi, "Paralyzed"))
        elif t == "Confused":
            lines.append(status_line(e, pi, "Confused"))

        # --- COIN (22) ---
        elif t == "Coin":
            face = "Heads" if e.get("head") else "Tails"
            lines.append(f"{player_label(pi)} flips a coin: {face}.")

        # --- RESULT (23) ---
        elif t == "Result":
            # Per docs (LogType.RESULT), this event does NOT carry a meaningful
            # playerIndex -- result and reason are absolute, not relative to `pi`:
            #   result: 0 = Player 1 (index 0) wins, 1 = Player 2 (index 1) wins, 2 = draw
            #   reason: 1 = 0 prize cards, 2 = no deck, 3 = no Active Pokémon, 4 = card effect
            # The previous version of this script misread `result` as a win/loss/draw
            # flag *for* `pi`, which is `None` for this event type -- that's why every
            # match end printed "A player" regardless of who actually won.
            reason = e.get("reason")
            reason_text = {
                1: "0 Prize cards remaining (all Prize cards taken)",
                2: "Deck Out (0 cards remaining in deck)",
                3: "No Active Pokémon remaining",
                4: "Card effect",
            }.get(reason, f"Undocumented reason code ({reason})")

            res_code = e.get("result")
            if res_code == 0:
                outcome = f"{player_label(0)} wins, {player_label(1)} loses"
            elif res_code == 1:
                outcome = f"{player_label(1)} wins, {player_label(0)} loses"
            elif res_code == 2:
                outcome = "The match ends in a draw"
            else:
                outcome = f"Undocumented result code ({res_code})"

            lines.append(f"**Match Ended: {outcome} — reason: {reason_text}.**")

        else:
            # This only fires for a `type` string outside the documented LogType
            # enum (https://matsuoinstitute.github.io/cabt/api.html#api.LogType) --
            # e.g. if the engine adds a new log type in a future module version.
            details = {k: v for k, v in e.items() if k not in ("type", "playerIndex")}
            lines.append(f"[undocumented LogType: {t}] {player_label(pi)} — {details}")

        i += 1

    return lines

def board_snapshot(current: Optional[Dict[str, Any]], id_name: Dict[int, str]) -> str:
    if not current or "players" not in current:
        return ""

    out: List[str] = []
    for pi, p in enumerate(current["players"]):
        active = [f"{c['name']} (#{c.get('serial')})" for c in p.get("active", [])]
        bench = [f"{c['name']} (#{c.get('serial')})" for c in p.get("bench", [])]
        discard_n = len(p.get("discard", []))
        prize_n = len(p.get("prize", []))
        hand_n = p.get("handCount", len(p.get("hand", [])))
        deck_n = p.get("deckCount", len(p.get("deck", [])))

        out.append(f"- **Player {pi + 1}**: Active: {', '.join(active) if active else '*(empty)*'}; Bench: {', '.join(bench) if bench else '*(empty)*'}; Hand: {hand_n}; Deck: {deck_n}; Discard: {discard_n}; Prizes left: {prize_n}")
    return "\n".join(out)

def decklist_markdown(current: Optional[Dict[str, Any]]) -> str:
    if not current or "players" not in current:
        return ""
    blocks: List[str] = []
    for pi, p in enumerate(current["players"]):
        deck = p.get("deck", [])
        if not deck:
            continue
        cnt = collections.Counter(c["name"] for c in deck)
        rows = "\n".join(f"| {c} | {name} |" for name, c in sorted(cnt.items(), key=lambda x: -x[1]))
        blocks.append(f"**Player {pi + 1}** ({len(deck)} cards)\n\n| Qty | Card |\n|---|---|\n{rows}")
    return "\n\n".join(blocks)

def process_replay(input_data: ReplayParseInput) -> ReplayParseOutput:
    data = input_data.raw_json
    id_name = build_id_name_map(data)
    viz = find_setup_visualize(data)

    title = data.get("title", data.get("name", "TCG Match"))
    match_id = data.get("id", "unknown")
    module_version = data.get("module_version", "unknown")

    rewards = data.get("rewards", [])
    statuses = data.get("statuses", [])

    md: List[str] = []
    md.append(f"# {title} — Step-by-Step Replay Breakdown\n")
    md.append(f"**Match ID:** `{match_id}`  ")
    md.append(f"**Engine:** `{data.get('name', 'unknown')}` (module v{module_version})  ")
    
    team_names = data.get("info", {}).get("TeamNames", [])
    def get_player_name(idx):
        if team_names and idx < len(team_names):
            return team_names[idx]
        return f"Player {idx+1}"
    
    n_agents = None
    try:
        if team_names and len(team_names) >= 2:
            n_agents = f"{team_names[0]} vs {team_names[1]}"
        else:
            n_agents = data.get("specification", {}).get("agents", [])
    except Exception:
        pass
    if n_agents:
        md.append(f"**Agents:** {n_agents}")

    if rewards:
        clean_rewards = [r if r is not None else 0 for r in rewards]
        max_reward = max(clean_rewards)
        if max_reward <= 0:
            winner_text = "None (Tie / Error)"
        else:
            winners = [get_player_name(i) for i, r in enumerate(clean_rewards) if r == max_reward]
            winner_text = ", ".join(winners) if winners else "None"
        md.append(f"**Winner:** {winner_text}")
        rewards_text = ", ".join([f"{get_player_name(i)}: {r}" for i, r in enumerate(rewards)])
        md.append(f"**Rewards:** {rewards_text}")
    
    steps = data.get("steps", [])
    total_steps = len(steps)
    md.append(f"**Total Steps:** {total_steps}")
    
    player_step_counts = {}
    for step in steps:
        if isinstance(step, list):
            for i, agent_step in enumerate(step):
                # Use `is not None` rather than truthiness -- action index 0 is a
                # valid, legitimate action and was previously undercounted because
                # `agent_step.get("action")` evaluates falsy for 0.
                if agent_step is not None and agent_step.get("action") is not None:
                    pname = get_player_name(i)
                    player_step_counts[pname] = player_step_counts.get(pname, 0) + 1
    
    if player_step_counts:
        step_counts_text = ", ".join([f"{k}: {v}" for k, v in player_step_counts.items()])
        md.append(f"**Agent Steps:** {step_counts_text}")
    
    md.append("\n")

    if rewards or statuses:
        md.append("## Result\n")
        n = max(len(rewards), len(statuses))
        header = "| | " + " | ".join(f"Player {i+1}" for i in range(n)) + " |"
        sep = "|---" * (n + 1) + "|"
        status_row = "| Final status | " + " | ".join(str(statuses[i]) if i < len(statuses) else "" for i in range(n)) + " |"
        reward_row = "| Reward | " + " | ".join(str(rewards[i]) if i < len(rewards) else "" for i in range(n)) + " |"
        md.append(header)
        md.append(sep)
        md.append(status_row)
        md.append(reward_row)
        md.append("")

    if not viz:
        md.append("_No detailed engine log ('visualize') was found in this replay file._")
        return ReplayParseOutput(markdown="\n".join(md))

    first_current = viz[0].get("current")
    decks_md = decklist_markdown(first_current)
    if decks_md:
        md.append("## Decklists\n")
        md.append(decks_md)
        md.append("")

    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for entry in viz:
        cur = entry.get("current") or {}
        turn = cur.get("turn")
        label = "Setup" if turn == 0 else (f"Turn {turn}" if turn is not None else "Unlabeled")
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append(entry)

    md.append("## Match Log\n")
    for label, entries in groups:
        md.append(f"### {label}\n")
        any_lines = False
        for entry in entries:
            lines = decode_entry_logs(entry.get("logs") or [], id_name)
            for line in lines:
                if line.startswith("###TURNDIV###"):
                    md.append("")
                    md.append(f"**{line[len('###TURNDIV###'):]}**")
                    md.append("")
                elif line.startswith("  -"):
                    md.append(line)
                else:
                    md.append(f"- {line}")
                any_lines = True
        if not any_lines:
            md.append("*(no new events at this step)*")

        snap = board_snapshot(entries[-1].get("current"), id_name)
        if snap:
            md.append("\n**Board after this point:**\n")
            md.append(snap)
        md.append("")

    return ReplayParseOutput(markdown="\n".join(md))

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 replay_to_md.py path/to/replay.json [output.md]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = in_path.with_name(in_path.stem + "_replay.md")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    parse_input = ReplayParseInput(raw_json=data)
    parse_output = process_replay(parse_input)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(parse_output.markdown)

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()