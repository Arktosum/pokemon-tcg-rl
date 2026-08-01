"""
greedy_bot.py — GreedyBot: Deck-agnostic score-based rule agent.
TITAN V5.0 Rule-Based Bot Curriculum.

Philosophy: Maximize immediate impact. No planning, no search.
Scores encode turn ordering (evolve > attach > play > attack > end),
mirroring the proven pattern from the Dragapult community notebook.
"""

import os
import sys
import traceback
from typing import Optional

# Add baseline agent dir to sys.path so cg package is importable
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
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectContext,
    SelectData,
    all_attack,
    to_observation_class,
)

from bot_common import (
    get_card,
    hp_ratio,
    energy_count,
    load_card_table,
    read_deck_csv,
    safe_select,
    unstruct,
)


# ---------------------------------------------------------------------------
# Attack damage lookup (cached)
# ---------------------------------------------------------------------------
_attack_damage: Optional[dict[int, int]] = None


def _load_attack_damage() -> dict[int, int]:
    """Build {attackId: damage} lookup from engine."""
    global _attack_damage
    if _attack_damage is None:
        attacks = all_attack()
        _attack_damage = {a.attackId: a.damage for a in attacks}
    return _attack_damage


# ---------------------------------------------------------------------------
# Scoring constants (turn ordering)
# ---------------------------------------------------------------------------
SCORE_EVOLVE = 50000
SCORE_ABILITY = 25000
SCORE_ATTACH = 30000
SCORE_PLAY_POKEMON = 18000
SCORE_PLAY_SUPPORTER = 20000
SCORE_PLAY_ITEM = 15000
SCORE_PLAY_ENERGY = 5000
SCORE_ATTACK_BASE = 1000
SCORE_END = 10
SCORE_YES = 1
SCORE_NO = 0
SCORE_DEFAULT = 0
SCORE_SKIP = -1


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------
def agent(obs_dict: dict) -> list[int]:
    """GreedyBot agent.

    Scores every legal option and returns the top maxCount indices.
    Devil's Advocate: wraps in try/except for robustness.
    """
    try:
        obs = to_observation_class(unstruct(obs_dict))

        # Initial deck selection
        if obs.select is None:
            return read_deck_csv()

        card_table = load_card_table()
        attack_damage = _load_attack_damage()

        state = obs.current
        select = obs.select
        context = select.context
        my_index = state.yourIndex

        scores = []
        for o in select.option:
            scores.append(_score_option(o, obs, context, my_index, card_table, attack_damage))

        return safe_select(scores, select)
    except Exception:
        # Devil's Advocate: return safe default (first option)
        traceback.print_exc()
        return [0]


# ---------------------------------------------------------------------------
# Option scoring
# ---------------------------------------------------------------------------
def _score_option(
    o: Option,
    obs: Observation,
    context: SelectContext,
    my_index: int,
    card_table: dict[int, CardData],
    attack_damage: dict[int, int],
) -> float:
    """Score a single option. Higher = more preferred."""
    try:
        if o.type == OptionType.YES:
            return SCORE_YES

        if o.type == OptionType.NO:
            return SCORE_NO

        if o.type == OptionType.NUMBER:
            return float(o.number) if o.number is not None else SCORE_DEFAULT

        if o.type == OptionType.END:
            return SCORE_END

        if o.type == OptionType.ATTACK:
            return _score_attack(o, attack_damage)

        if o.type == OptionType.EVOLVE:
            return _score_evolve(o, obs, my_index)

        if o.type == OptionType.ATTACH:
            return _score_attach(o, obs, my_index)

        if o.type == OptionType.PLAY:
            return _score_play(o, obs, my_index, card_table)

        if o.type == OptionType.ABILITY:
            return SCORE_ABILITY

        if o.type == OptionType.RETREAT:
            return SCORE_SKIP  # GreedyBot never retreats

        if o.type == OptionType.CARD:
            return _score_card(o, obs, context, my_index, card_table)

        if o.type in (OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
            return _score_attached_card(o, obs, context, my_index)

        if o.type == OptionType.SKILL:
            return 100.0

        if o.type == OptionType.SPECIAL_CONDITION:
            return SCORE_DEFAULT

    except Exception:
        # Devil's Advocate: any parsing error → safe default
        return SCORE_DEFAULT

    return SCORE_DEFAULT


# ---------------------------------------------------------------------------
# Attack scoring
# ---------------------------------------------------------------------------
def _score_attack(o: Option, attack_damage: dict[int, int]) -> float:
    """Score an attack by its base damage. Higher damage = higher score."""
    if o.attackId is None:
        return SCORE_ATTACK_BASE
    damage = attack_damage.get(o.attackId, 0)
    return SCORE_ATTACK_BASE + damage


# ---------------------------------------------------------------------------
# Evolve scoring
# ---------------------------------------------------------------------------
def _score_evolve(o: Option, obs: Observation, my_index: int) -> float:
    """Score evolution. Always high priority (evolve first)."""
    pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
    bonus = energy_count(pokemon) if pokemon is not None else 0
    return SCORE_EVOLVE + bonus


# ---------------------------------------------------------------------------
# Attach scoring
# ---------------------------------------------------------------------------
def _score_attach(o: Option, obs: Observation, my_index: int) -> float:
    """Score energy/tool attachment. Prefer active Pokemon with 0 energy."""
    pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
    if pokemon is None:
        return SCORE_SKIP

    ec = energy_count(pokemon)
    is_active = o.inPlayArea == AreaType.ACTIVE

    # Prefer active with 0 energy, then benched with 0 energy
    if is_active:
        if ec == 0:
            return SCORE_ATTACH + 500
        return SCORE_ATTACH + 100
    else:
        if ec == 0:
            return SCORE_ATTACH + 200
        return SCORE_ATTACH


# ---------------------------------------------------------------------------
# Play scoring
# ---------------------------------------------------------------------------
def _score_play(o: Option, obs: Observation, my_index: int, card_table: dict[int, CardData]) -> float:
    """Score playing a card from hand by card type."""
    card = get_card(obs, AreaType.HAND, o.index, my_index)
    if card is None:
        return SCORE_SKIP

    data = card_table.get(card.id)
    if data is None:
        return SCORE_PLAY_ITEM  # Unknown card, mild preference

    if data.cardType == CardType.SUPPORTER:
        return SCORE_PLAY_SUPPORTER
    elif data.cardType == CardType.POKEMON:
        return SCORE_PLAY_POKEMON
    elif data.cardType == CardType.ITEM:
        return SCORE_PLAY_ITEM
    elif data.cardType == CardType.TOOL:
        return SCORE_PLAY_ITEM
    elif data.cardType == CardType.BASIC_ENERGY:
        return SCORE_PLAY_ENERGY
    elif data.cardType == CardType.SPECIAL_ENERGY:
        return SCORE_PLAY_ENERGY
    elif data.cardType == CardType.STADIUM:
        return SCORE_PLAY_ITEM

    return SCORE_PLAY_ITEM


# ---------------------------------------------------------------------------
# Card selection scoring (context-dependent)
# ---------------------------------------------------------------------------
def _score_card(
    o: Option,
    obs: Observation,
    context: SelectContext,
    my_index: int,
    card_table: dict[int, CardData],
) -> float:
    """Score a CARD option based on the selection context."""
    card = get_card(obs, o.area, o.index, o.playerIndex)
    if card is None:
        return SCORE_DEFAULT

    # Extract Pokemon info if applicable
    ec = 0
    hp = 0
    if isinstance(card, Pokemon):
        ec = energy_count(card)
        hp = card.hp

    # --- Setup: pick highest HP Basic ---
    if context in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
        return float(hp)

    # --- Switch / To Active: promote tankiest ---
    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        if o.playerIndex == my_index:
            return float(hp) + ec * 100
        else:
            # Targeting opponent for Boss Orders-like effects
            return float(hp)

    # --- To Bench / To Hand: recover valuable cards ---
    if context in (SelectContext.TO_BENCH, SelectContext.TO_HAND):
        data = card_table.get(card.id) if not isinstance(card, Pokemon) else card_table.get(card.id)
        if data is not None:
            if data.cardType == CardType.POKEMON:
                return 100.0
            if data.cardType == CardType.BASIC_ENERGY:
                return 50.0
        return 10.0

    # --- Attach From: prefer Pokemon with 0 energy ---
    if context == SelectContext.ATTACH_FROM:
        if isinstance(card, Pokemon):
            if ec == 0:
                return 1000.0
            return 100.0
        return SCORE_DEFAULT

    # --- Damage Counter: target closest to KO ---
    if context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
        if hp > 0:
            return 100000.0 - 10.0 * hp
        return SCORE_DEFAULT

    # --- Damage: target opponent Pokemon ---
    if context == SelectContext.DAMAGE:
        if hp > 0:
            return 100000.0 - 10.0 * hp
        return SCORE_DEFAULT

    # --- Heal: heal closest to dying ---
    if context == SelectContext.HEAL:
        if hp > 0:
            return 100000.0 - 10.0 * hp
        return SCORE_DEFAULT

    # --- Remove Damage Counter: heal own Pokemon ---
    if context == SelectContext.REMOVE_DAMAGE_COUNTER:
        if hp > 0:
            return 100000.0 - 10.0 * hp
        return SCORE_DEFAULT

    # --- Discard: discard lowest value ---
    if context == SelectContext.DISCARD:
        return -float(hp)

    # --- To Deck / To Deck Bottom: low priority ---
    if context in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
        return 1.0

    # --- Not Move: keep in place ---
    if context == SelectContext.NOT_MOVE:
        return 1.0

    # --- Default ---
    return SCORE_DEFAULT


# ---------------------------------------------------------------------------
# Attached card scoring (TOOL_CARD, ENERGY_CARD, ENERGY)
# ---------------------------------------------------------------------------
def _score_attached_card(
    o: Option,
    obs: Observation,
    context: SelectContext,
    my_index: int,
) -> float:
    """Score selecting an attached tool/energy card."""
    # For discarding attached cards (e.g., Crushing Hammer, retreat cost)
    if context in (SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_TOOL_CARD,
                   SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
        return 10.0  # Mild preference to discard

    # For switching/replacing energy
    if context in (SelectContext.SWITCH_ENERGY_CARD, SelectContext.SWITCH_ENERGY,
                   SelectContext.DISCARD_ENERGY):
        return 10.0

    return 10.0