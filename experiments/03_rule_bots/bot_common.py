"""
bot_common.py — Shared utilities for all rule-based bots.
TITAN V5.0 Rule-Based Bot Curriculum.

Provides deck-agnostic helpers used by GreedyBot, TacticalBot, and SearchBot:
- read_deck_csv(): Load deck.csv (60 card IDs) with fallback paths.
- get_card(): Safe card extraction from any AreaType.
- load_card_table(): Build {cardId: CardData} lookup from engine.
- prize_value(): Calculate prize cards on KO (megaEx=3, ex=2, else=1).
- hp_ratio(): Current HP / Max HP.
- energy_count(): Number of energies attached.
- safe_select(): Pick top maxCount indices from scores, respecting minCount.
- estimate_damage(): Rough damage estimate for an attack given attacker data.
"""

import os
import sys
from typing import Optional

# Add baseline agent dir to sys.path so cg package is importable
# Handle both __file__ context and kaggle_environments exec context
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # kaggle_environments exec context — __file__ not available.
    # Handle case where cwd is already the bot directory or the repo root.
    _cwd = os.getcwd()
    if os.path.basename(_cwd) == '03_rule_bots':
        _this_dir = _cwd
    else:
        _this_dir = os.path.join(_cwd, 'experiments', '03_rule_bots')
_agent_dir = os.path.abspath(os.path.join(_this_dir, '..', '01_baseline', 'agent'))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from cg.api import (
    AreaType,
    Card,
    CardData,
    CardType,
    EnergyType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectContext,
    SelectData,
    all_card_data,
    to_observation_class,
)


def unstruct(obj):
    """Recursively convert Struct (dict subclass) objects to plain dicts/lists.

    kaggle_environments' structify() wraps all dicts in a Struct class.
    While Struct is a dict subclass, to_dataclass can misbehave with certain
    nested Structures in Python 3.13. This ensures clean conversion.
    """
    if isinstance(obj, dict):
        return {k: unstruct(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [unstruct(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Deck loading
# ---------------------------------------------------------------------------
def read_deck_csv() -> list[int]:
    """Read deck.csv and return 60 card IDs.

    Searches multiple fallback paths for local dev and Kaggle submission.
    """
    candidates = [
        "deck.csv",
        os.path.join(_this_dir, "deck.csv"),
        os.path.join(os.getcwd(), "experiments", "03_rule_bots", "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
            deck = []
            for line in lines:
                line = line.strip()
                if line:
                    deck.append(int(line))
                if len(deck) == 60:
                    break
            if len(deck) == 60:
                return deck
    raise FileNotFoundError("deck.csv not found in any expected location.")


# ---------------------------------------------------------------------------
# Card table
# ---------------------------------------------------------------------------
_card_table: Optional[dict[int, CardData]] = None


def load_card_table() -> dict[int, CardData]:
    """Build and cache a {cardId: CardData} lookup from the engine."""
    global _card_table
    if _card_table is None:
        all_cards = all_card_data()
        _card_table = {c.cardId: c for c in all_cards}
    return _card_table


# ---------------------------------------------------------------------------
# Board extraction
# ---------------------------------------------------------------------------
def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Optional[Pokemon | Card]:
    """Safely extract a Card or Pokemon from a specific zone.

    Mirrors the helper used in both community notebooks.
    """
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        deck = obs.select.deck
        if deck is not None and 0 <= index < len(deck):
            return deck[index]
        return None
    if area == AreaType.HAND:
        hand = ps.hand
        if hand is not None and 0 <= index < len(hand):
            return hand[index]
        return None
    if area == AreaType.DISCARD:
        if 0 <= index < len(ps.discard):
            return ps.discard[index]
        return None
    if area == AreaType.ACTIVE:
        if 0 <= index < len(ps.active):
            return ps.active[index]
        return None
    if area == AreaType.BENCH:
        if 0 <= index < len(ps.bench):
            return ps.bench[index]
        return None
    if area == AreaType.PRIZE:
        if 0 <= index < len(ps.prize):
            return ps.prize[index]
        return None
    if area == AreaType.STADIUM:
        if 0 <= index < len(obs.current.stadium):
            return obs.current.stadium[index]
        return None
    if area == AreaType.LOOKING:
        looking = obs.current.looking
        if looking is not None and 0 <= index < len(looking):
            return looking[index]
        return None
    return None


# ---------------------------------------------------------------------------
# Pokemon heuristics
# ---------------------------------------------------------------------------
def hp_ratio(pokemon: Optional[Pokemon]) -> float:
    """Return current HP / max HP. Returns 0.0 for None."""
    if pokemon is None:
        return 0.0
    if pokemon.maxHp <= 0:
        return 0.0
    return pokemon.hp / pokemon.maxHp


def energy_count(pokemon: Optional[Pokemon]) -> int:
    """Return number of energies attached. Returns 0 for None."""
    if pokemon is None:
        return 0
    return len(pokemon.energies)


def prize_value(pokemon: Optional[Pokemon], card_table: dict[int, CardData]) -> int:
    """Calculate how many Prize cards a Pokemon yields on KO.

    megaEx = 3, ex = 2, else = 1.
    Subtracts 1 for Legacy Energy (id=12) and Lillie's Pearl (id=1172) on Lillie cards.
    """
    if pokemon is None:
        return 0
    data = card_table.get(pokemon.id)
    if data is None:
        return 1
    count = 3 if data.megaEx else 2 if data.ex else 1
    # Legacy Energy reduces prize count
    for card in pokemon.energyCards:
        if card.id == 12:
            count -= 1
    # Lillie's Pearl reduces prize count for Lillie cards
    for tool in pokemon.tools:
        if tool.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def is_ex(pokemon: Optional[Pokemon], card_table: dict[int, CardData]) -> bool:
    """Check if a Pokemon is an ex or Mega ex (gives 2+ prizes on KO)."""
    if pokemon is None:
        return False
    data = card_table.get(pokemon.id)
    if data is None:
        return False
    return data.ex or data.megaEx


def is_basic(pokemon: Optional[Pokemon], card_table: dict[int, CardData]) -> bool:
    """Check if a Pokemon is a Basic (can be played directly to bench)."""
    if pokemon is None:
        return False
    data = card_table.get(pokemon.id)
    if data is None:
        return False
    return data.basic


# ---------------------------------------------------------------------------
# Attack estimation
# ---------------------------------------------------------------------------
def estimate_attack_damage(attack_id: int, card_table: dict[int, CardData]) -> int:
    """Rough damage estimate for an attack by ID.

    Looks up the attack in all card data. Returns 0 if not found.
    This is a rough heuristic — actual damage depends on card effects.
    """
    # We don't have a direct attack_id -> damage map, but we can search
    # all cards' attacks. This is cached after first call.
    if not hasattr(estimate_attack_damage, "_attack_map"):
        attack_map = {}
        for card in card_table.values():
            for aid in card.attacks:
                if aid not in attack_map:
                    # We don't have Attack data directly in CardData,
                    # so we use a rough heuristic based on card HP
                    attack_map[aid] = 0
        estimate_attack_damage._attack_map = attack_map
    return estimate_attack_damage._attack_map.get(attack_id, 0)


def estimate_opponent_damage(pokemon: Optional[Pokemon], card_table: dict[int, CardData]) -> int:
    """Rough estimate of how much damage an opponent's Pokemon can deal next turn.

    Heuristic: energy_count * 40 as a baseline, since most attacks need 1-3 energy.
    """
    if pokemon is None:
        return 0
    ec = energy_count(pokemon)
    # Assume they can attach 1 more energy next turn
    ec_next = ec + 1
    # Rough baseline: more energy = more damage potential
    return ec_next * 40


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------
def safe_select(scores: list[float], select: SelectData) -> list[int]:
    """Pick the top maxCount option indices from scores, respecting minCount.

    Follows the pattern from the Dragapult community notebook:
    - Sort options by score descending.
    - Select up to maxCount options.
    - If score is negative and we can skip (minCount satisfied), skip it.
    - Never return duplicate indices.
    - Always return at least minCount indices.
    """
    n = len(scores)
    if n == 0:
        return []

    # Sort indices by score descending
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)

    output = []
    for rank, idx in enumerate(sorted_indices):
        if len(output) >= select.maxCount:
            break
        score = scores[idx]
        # If score is negative, skip unless we need to satisfy minCount
        if score < 0 and rank >= select.minCount:
            continue
        output.append(idx)

    # Ensure we return at least minCount
    while len(output) < select.minCount and len(output) < n:
        for idx in sorted_indices:
            if idx not in output:
                output.append(idx)
                break
        else:
            break

    return output


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------
def my_board(obs: Observation) -> list[Optional[Pokemon]]:
    """Return [active, bench1, bench2, ...] for the current player."""
    me = obs.current.players[obs.current.yourIndex]
    board = list(me.active) + list(me.bench)
    return board


def opponent_board(obs: Observation) -> list[Optional[Pokemon]]:
    """Return [active, bench1, bench2, ...] for the opponent."""
    op = obs.current.players[1 - obs.current.yourIndex]
    board = list(op.active) + list(op.bench)
    return board


def my_prize_count(obs: Observation) -> int:
    """Number of prize cards remaining for us (6 = start, 0 = won)."""
    me = obs.current.players[obs.current.yourIndex]
    return len(me.prize)


def opponent_prize_count(obs: Observation) -> int:
    """Number of prize cards remaining for opponent."""
    op = obs.current.players[1 - obs.current.yourIndex]
    return len(op.prize)