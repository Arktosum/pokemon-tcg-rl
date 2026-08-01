"""
tactical_bot.py — TacticalBot: Prize-aware rule agent with threat assessment.
TITAN V5.0 Rule-Based Bot Curriculum.

Philosophy: Aware of prize trade economics. Protects high-value Pokemon (ex/Mega ex).
Retreats strategically when HP is low. Weighs damage vs self-risk.

Extends GreedyBot with:
- Retreat logic: If active HP < 30% and bench has healthier Pokemon, retreat.
- ex protection: Penalize ex Pokemon as active (gives 2+ prizes on KO).
- Attack risk: Check if opponent can KO our active next turn.
- Prize-aware targeting: Weight DAMAGE_COUNTER by prize_value.
- Lethal detection: If KO takes last prize, boost to 999999.
- Deck conservation: If deckCount < 10, penalize draw supporters.
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
    prize_value,
    is_ex,
    is_basic,
    load_card_table,
    read_deck_csv,
    safe_select,
    my_board,
    opponent_board,
    my_prize_count,
    opponent_prize_count,
    estimate_opponent_damage,
    unstruct,
)
import greedy_bot


# ---------------------------------------------------------------------------
# Attack damage lookup (reuse from greedy_bot)
# ---------------------------------------------------------------------------
def _load_attack_damage() -> dict[int, int]:
    return greedy_bot._load_attack_damage()


# ---------------------------------------------------------------------------
# Tactical scoring constants
# ---------------------------------------------------------------------------
SCORE_RETREAT = 40000
SCORE_LETHAL = 999999
EX_PENALTY = 5000
LOW_HP_THRESHOLD = 0.3
HEALTHY_HP_THRESHOLD = 0.5
LOW_DECK_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------
def agent(obs_dict: dict) -> list[int]:
    """TacticalBot agent.

    Scores every legal option with prize-trade awareness and returns top maxCount.
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

        # Pre-compute board state for tactical decisions
        me = state.players[my_index]
        op = state.players[1 - my_index]
        my_active = me.active[0] if len(me.active) > 0 else None
        op_active = op.active[0] if len(op.active) > 0 else None
        my_prizes = my_prize_count(obs)
        op_prizes = opponent_prize_count(obs)
        deck_count = me.deckCount

        # Check if we should retreat
        should_retreat = _should_retreat(my_active, me.bench, card_table)

        # Check if opponent can KO our active next turn
        op_threat = estimate_opponent_damage(op_active, card_table) if op_active else 0
        active_in_danger = my_active is not None and op_threat >= my_active.hp

        scores = []
        for o in select.option:
            # Start with GreedyBot's base score
            base_score = greedy_bot._score_option(o, obs, context, my_index, card_table, attack_damage)

            # Apply tactical overrides
            tactical_score = _apply_tactical_overrides(
                o, obs, context, my_index, card_table, attack_damage,
                my_active, op_active, my_prizes, op_prizes, deck_count,
                should_retreat, active_in_danger, base_score
            )
            scores.append(tactical_score)

        return safe_select(scores, select)
    except Exception:
        # Devil's Advocate: return safe default (first option)
        traceback.print_exc()
        return [0]


# ---------------------------------------------------------------------------
# Tactical override logic
# ---------------------------------------------------------------------------
def _apply_tactical_overrides(
    o: Option,
    obs: Observation,
    context: SelectContext,
    my_index: int,
    card_table: dict[int, CardData],
    attack_damage: dict[int, int],
    my_active: Optional[Pokemon],
    op_active: Optional[Pokemon],
    my_prizes: int,
    op_prizes: int,
    deck_count: int,
    should_retreat: bool,
    active_in_danger: bool,
    base_score: float,
) -> float:
    """Apply tactical adjustments on top of GreedyBot's base score."""

    # --- Retreat override ---
    if o.type == OptionType.RETREAT:
        if should_retreat:
            return SCORE_RETREAT
        return greedy_bot.SCORE_SKIP

    # --- ex protection for SWITCH / TO_ACTIVE / SETUP_ACTIVE ---
    if o.type == OptionType.CARD and context in (
        SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON
    ):
        if o.playerIndex == my_index:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if isinstance(card, Pokemon):
                # Penalize ex Pokemon as active
                if is_ex(card, card_table):
                    base_score -= EX_PENALTY
                # Boost if this Pokemon is a safe non-ex tank
                if not is_ex(card, card_table) and card.hp > 0:
                    base_score += 500

    # --- Lethal detection for ATTACK ---
    if o.type == OptionType.ATTACK and op_active is not None:
        attack_dmg = _get_attack_damage(o, attack_damage)
        if attack_dmg > 0 and op_active.hp > 0 and attack_dmg >= op_active.hp:
            # This attack KOs the opponent's active
            prizes_gained = prize_value(op_active, card_table)
            if prizes_gained >= op_prizes:
                # This takes our last prize(s) = WIN
                return SCORE_LETHAL
            # Otherwise, boost by prize value
            base_score += prizes_gained * 1000

    # --- Attack risk assessment ---
    if o.type == OptionType.ATTACK and active_in_danger and not should_retreat:
        # If we're in danger but can't retreat, attacking is better than ending
        base_score += 500

    # --- Prize-aware DAMAGE_COUNTER targeting ---
    if o.type == OptionType.CARD and context in (
        SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY
    ):
        card = get_card(obs, o.area, o.index, o.playerIndex)
        if isinstance(card, Pokemon) and card.hp > 0:
            # Weight by prize value (prefer KOing ex/Mega ex)
            pv = prize_value(card, card_table)
            # Check if this counter would KO
            remain = obs.select.remainDamageCounter if obs.select else 0
            if remain * 10 >= card.hp:
                # This counter KOs the target
                if pv >= op_prizes:
                    return SCORE_LETHAL
                base_score += pv * 1000
            else:
                # Not lethal, but still weight by prize value
                base_score += pv * 100

    # --- Prize-aware HEAL targeting ---
    if o.type == OptionType.CARD and context == SelectContext.HEAL:
        card = get_card(obs, o.area, o.index, o.playerIndex)
        if isinstance(card, Pokemon) and card.hp > 0:
            # Prefer healing ex Pokemon (protect our investment)
            if is_ex(card, card_table):
                base_score += 2000

    # --- Deck conservation ---
    if o.type == OptionType.PLAY and deck_count < LOW_DECK_THRESHOLD:
        card = get_card(obs, AreaType.HAND, o.index, my_index)
        if card is not None:
            data = card_table.get(card.id)
            # Penalize playing draw supporters when deck is low
            if data is not None and data.cardType == CardType.SUPPORTER:
                base_score -= 10000

    return base_score


# ---------------------------------------------------------------------------
# Helper: should we retreat?
# ---------------------------------------------------------------------------
def _should_retreat(
    my_active: Optional[Pokemon],
    my_bench: list,
    card_table: dict[int, CardData],
) -> bool:
    """Check if we should retreat the active Pokemon.

    Conditions:
    - Active HP < 30%
    - Bench has a Pokemon with HP > 50%
    """
    if my_active is None:
        return False

    if hp_ratio(my_active) >= LOW_HP_THRESHOLD:
        return False

    # Check bench for a healthier Pokemon
    for bench_poke in my_bench:
        if bench_poke is not None and hp_ratio(bench_poke) > HEALTHY_HP_THRESHOLD:
            return True

    return False


# ---------------------------------------------------------------------------
# Helper: get attack damage
# ---------------------------------------------------------------------------
def _get_attack_damage(o: Option, attack_damage: dict[int, int]) -> int:
    """Get the base damage for an attack option."""
    if o.attackId is None:
        return 0
    return attack_damage.get(o.attackId, 0)
