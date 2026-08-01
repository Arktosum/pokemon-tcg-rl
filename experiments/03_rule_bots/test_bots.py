"""
test_bots.py — Rigorous unit tests for rule-based bots.
TITAN V5.0 Rule-Based Bot Curriculum.

Tests cover:
- bot_common: safe_select, hp_ratio, energy_count, prize_value, is_ex
- greedy_bot: agent deck selection, score_option for all OptionTypes
- tactical_bot: _should_retreat, lethal detection, ex protection
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from typing import Optional

# Add this dir to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Add baseline agent dir for cg package
_agent_dir = os.path.abspath(os.path.join(_this_dir, '..', '01_baseline', 'agent'))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from cg.api import (
    AreaType, Card, CardData, CardType, EnergyType, Observation,
    Option, OptionType, Pokemon, SelectContext, SelectData,
    SelectType, State, PlayerState, SpecialConditionType,
)

from bot_common import (
    safe_select, hp_ratio, energy_count, prize_value, is_ex, get_card,
)


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------
def make_card(card_id=1, serial=1, player_index=0):
    return Card(id=card_id, serial=serial, playerIndex=player_index)

def make_pokemon(card_id=1, serial=1, hp=100, max_hp=100,
                 appear_this_turn=False, energies=None, energy_cards=None,
                 tools=None, pre_evolution=None):
    return Pokemon(
        id=card_id, serial=serial, hp=hp, maxHp=max_hp,
        appearThisTurn=appear_this_turn,
        energies=energies or [], energyCards=energy_cards or [],
        tools=tools or [], preEvolution=pre_evolution or [],
    )

def make_card_data(card_id=1, name="TestCard", card_type=CardType.POKEMON,
                   ex=False, mega_ex=False, basic=True, stage1=False,
                   stage2=False, hp=100):
    return CardData(
        cardId=card_id, name=name, cardType=card_type, retreatCost=1,
        hp=hp, weakness=None, resistance=None, energyType=EnergyType.COLORLESS,
        basic=basic, stage1=stage1, stage2=stage2, ex=ex, megaEx=mega_ex,
        tera=False, aceSpec=False, evolvesFrom=None, skills=[], attacks=[],
    )

def make_option(opt_type=OptionType.END, number=None, area=None, index=None,
                player_index=None, tool_index=None, energy_index=None,
                count=None, in_play_area=None, in_play_index=None,
                attack_id=None, card_id=None, serial=None,
                special_condition_type=None):
    return Option(
        type=opt_type, number=number, area=area, index=index,
        playerIndex=player_index, toolIndex=tool_index,
        energyIndex=energy_index, count=count, inPlayArea=in_play_area,
        inPlayIndex=in_play_index, attackId=attack_id, cardId=card_id,
        serial=serial, specialConditionType=special_condition_type,
    )

def make_select_data(options=None, context=SelectContext.MAIN,
                      min_count=1, max_count=1, remain_damage_counter=0,
                      remain_energy_cost=0, deck=None,
                      context_card=None, effect=None):
    return SelectData(
        type=SelectType.MAIN, context=context, minCount=min_count,
        maxCount=max_count, remainDamageCounter=remain_damage_counter,
        remainEnergyCost=remain_energy_cost, option=options or [],
        deck=deck, contextCard=context_card, effect=effect,
    )

def make_mock_card_table(*card_datas):
    return {cd.cardId: cd for cd in card_datas}


# ---------------------------------------------------------------------------
# Tests: safe_select
# ---------------------------------------------------------------------------
class TestSafeSelect:
    def test_empty_scores_returns_empty(self):
        """safe_select with empty scores returns empty list."""
        select = make_select_data(options=[], min_count=0, max_count=1)
        assert safe_select([], select) == []

    def test_all_negative_min_zero_returns_empty(self):
        """All negative scores with minCount=0 returns empty."""
        select = make_select_data(min_count=0, max_count=3)
        assert safe_select([-1.0, -2.0, -3.0], select) == []

    def test_min_count_enforcement(self):
        """minCount=1 with all negative returns 1 item (highest)."""
        select = make_select_data(min_count=1, max_count=3)
        result = safe_select([-1.0, -2.0, -3.0], select)
        assert len(result) == 1
        assert result[0] == 0

    def test_max_count_cap(self):
        """Returns at most maxCount items."""
        select = make_select_data(min_count=0, max_count=2)
        result = safe_select([10.0, 20.0, 30.0], select)
        assert len(result) == 2
        assert result == [2, 1]

    def test_ordering_descending(self):
        """Returns indices in descending score order."""
        select = make_select_data(min_count=0, max_count=5)
        result = safe_select([5.0, 3.0, 8.0, 1.0, 6.0], select)
        assert result == [2, 4, 0, 1, 3]

    def test_no_duplicates(self):
        """No duplicate indices in output."""
        select = make_select_data(min_count=0, max_count=3)
        result = safe_select([1.0, 2.0, 3.0], select)
        assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# Tests: hp_ratio
# ---------------------------------------------------------------------------
class TestHpRatio:
    def test_none_returns_zero(self):
        assert hp_ratio(None) == 0.0

    def test_zero_max_hp_returns_zero(self):
        poke = make_pokemon(hp=50, max_hp=0)
        assert hp_ratio(poke) == 0.0

    def test_half_hp(self):
        poke = make_pokemon(hp=50, max_hp=100)
        assert hp_ratio(poke) == 0.5

    def test_full_hp(self):
        poke = make_pokemon(hp=100, max_hp=100)
        assert hp_ratio(poke) == 1.0

    def test_low_hp(self):
        poke = make_pokemon(hp=20, max_hp=100)
        assert hp_ratio(poke) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Tests: energy_count
# ---------------------------------------------------------------------------
class TestEnergyCount:
    def test_none_returns_zero(self):
        assert energy_count(None) == 0

    def test_no_energies(self):
        poke = make_pokemon(energies=[])
        assert energy_count(poke) == 0

    def test_three_energies(self):
        poke = make_pokemon(energies=[EnergyType.FIRE, EnergyType.WATER, EnergyType.GRASS])
        assert energy_count(poke) == 3


# ---------------------------------------------------------------------------
# Tests: prize_value
# ---------------------------------------------------------------------------
class TestPrizeValue:
    def test_none_returns_zero(self):
        assert prize_value(None, {}) == 0

    def test_normal_pokemon(self):
        poke = make_pokemon(card_id=1)
        table = make_mock_card_table(make_card_data(card_id=1, ex=False, mega_ex=False))
        assert prize_value(poke, table) == 1

    def test_ex_pokemon(self):
        poke = make_pokemon(card_id=2)
        table = make_mock_card_table(make_card_data(card_id=2, ex=True, mega_ex=False))
        assert prize_value(poke, table) == 2

    def test_mega_ex_pokemon(self):
        poke = make_pokemon(card_id=3)
        table = make_mock_card_table(make_card_data(card_id=3, ex=True, mega_ex=True))
        assert prize_value(poke, table) == 3

    def test_legacy_energy_reduces_prize(self):
        poke = make_pokemon(card_id=2, energy_cards=[make_card(card_id=12)])
        table = make_mock_card_table(make_card_data(card_id=2, ex=True, mega_ex=False))
        assert prize_value(poke, table) == 1

    def test_prize_never_negative(self):
        poke = make_pokemon(card_id=3, energy_cards=[make_card(card_id=12), make_card(card_id=12), make_card(card_id=12)])
        table = make_mock_card_table(make_card_data(card_id=3, ex=True, mega_ex=True))
        assert prize_value(poke, table) == 0


# ---------------------------------------------------------------------------
# Tests: is_ex
# ---------------------------------------------------------------------------
class TestIsEx:
    def test_none_returns_false(self):
        assert is_ex(None, {}) == False

    def test_normal_returns_false(self):
        poke = make_pokemon(card_id=1)
        table = make_mock_card_table(make_card_data(card_id=1, ex=False, mega_ex=False))
        assert is_ex(poke, table) == False

    def test_ex_returns_true(self):
        poke = make_pokemon(card_id=2)
        table = make_mock_card_table(make_card_data(card_id=2, ex=True, mega_ex=False))
        assert is_ex(poke, table) == True

    def test_mega_ex_returns_true(self):
        poke = make_pokemon(card_id=3)
        table = make_mock_card_table(make_card_data(card_id=3, ex=False, mega_ex=True))
        assert is_ex(poke, table) == True


# ---------------------------------------------------------------------------
# Tests: greedy_bot scoring
# ---------------------------------------------------------------------------
class TestGreedyScoring:
    def test_score_attack_with_damage(self):
        """Attack with known damage gets base + damage."""
        from greedy_bot import _score_attack
        attack_damage = {10: 50}
        o = make_option(opt_type=OptionType.ATTACK, attack_id=10)
        assert _score_attack(o, attack_damage) == 1050

    def test_score_attack_unknown_id(self):
        """Attack with unknown attackId gets base score only."""
        from greedy_bot import _score_attack
        o = make_option(opt_type=OptionType.ATTACK, attack_id=999)
        assert _score_attack(o, {10: 50}) == 1000

    def test_score_attack_none_id(self):
        """Attack with None attackId gets base score."""
        from greedy_bot import _score_attack
        o = make_option(opt_type=OptionType.ATTACK, attack_id=None)
        assert _score_attack(o, {}) == 1000

    def test_score_evolve_high_priority(self):
        """Evolve gets very high score (50000+)."""
        from greedy_bot import _score_evolve, SCORE_EVOLVE
        o = make_option(opt_type=OptionType.EVOLVE, in_play_area=AreaType.ACTIVE, in_play_index=0)
        with patch('greedy_bot.get_card', return_value=make_pokemon(energies=[EnergyType.FIRE])):
            score = _score_evolve(o, MagicMock(), 0)
        assert score >= SCORE_EVOLVE

    def test_score_attach_active_zero_energy(self):
        """Attach to active with 0 energy gets highest attach score."""
        from greedy_bot import _score_attach, SCORE_ATTACH
        o = make_option(opt_type=OptionType.ATTACH, in_play_area=AreaType.ACTIVE, in_play_index=0)
        with patch('greedy_bot.get_card', return_value=make_pokemon(energies=[])):
            score = _score_attach(o, MagicMock(), 0)
        assert score == SCORE_ATTACH + 500

    def test_score_attach_bench_with_energy(self):
        """Attach to benched Pokemon with energy gets base attach score."""
        from greedy_bot import _score_attach, SCORE_ATTACH
        o = make_option(opt_type=OptionType.ATTACH, in_play_area=AreaType.BENCH, in_play_index=0)
        with patch('greedy_bot.get_card', return_value=make_pokemon(energies=[EnergyType.FIRE])):
            score = _score_attach(o, MagicMock(), 0)
        assert score == SCORE_ATTACH

    def test_score_play_supporter(self):
        """Playing a supporter gets SCORE_PLAY_SUPPORTER."""
        from greedy_bot import _score_play, SCORE_PLAY_SUPPORTER
        o = make_option(opt_type=OptionType.PLAY, index=0)
        card = make_card(card_id=5)
        table = make_mock_card_table(make_card_data(card_id=5, card_type=CardType.SUPPORTER))
        with patch('greedy_bot.get_card', return_value=card):
            score = _score_play(o, MagicMock(), 0, table)
        assert score == SCORE_PLAY_SUPPORTER

    def test_score_play_pokemon(self):
        """Playing a Pokemon gets SCORE_PLAY_POKEMON."""
        from greedy_bot import _score_play, SCORE_PLAY_POKEMON
        o = make_option(opt_type=OptionType.PLAY, index=0)
        card = make_card(card_id=6)
        table = make_mock_card_table(make_card_data(card_id=6, card_type=CardType.POKEMON))
        with patch('greedy_bot.get_card', return_value=card):
            score = _score_play(o, MagicMock(), 0, table)
        assert score == SCORE_PLAY_POKEMON

    def test_score_yes(self):
        """YES option gets score 1."""
        from greedy_bot import _score_option, SCORE_YES
        o = make_option(opt_type=OptionType.YES)
        assert _score_option(o, MagicMock(), SelectContext.MAIN, 0, {}, {}) == SCORE_YES

    def test_score_end(self):
        """END option gets score 10."""
        from greedy_bot import _score_option, SCORE_END
        o = make_option(opt_type=OptionType.END)
        assert _score_option(o, MagicMock(), SelectContext.MAIN, 0, {}, {}) == SCORE_END

    def test_score_retreat_skip(self):
        """RETREAT option gets SCORE_SKIP in GreedyBot."""
        from greedy_bot import _score_option, SCORE_SKIP
        o = make_option(opt_type=OptionType.RETREAT)
        assert _score_option(o, MagicMock(), SelectContext.MAIN, 0, {}, {}) == SCORE_SKIP


# ---------------------------------------------------------------------------
# Tests: tactical_bot retreat logic
# ---------------------------------------------------------------------------
class TestTacticalRetreat:
    def test_should_retreat_low_hp_healthy_bench(self):
        """Low HP active + healthy bench Pokemon -> should retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=20, max_hp=100)
        bench = [make_pokemon(hp=80, max_hp=100)]
        assert _should_retreat(active, bench, {}) == True

    def test_should_retreat_high_hp(self):
        """High HP active -> should NOT retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=90, max_hp=100)
        bench = [make_pokemon(hp=80, max_hp=100)]
        assert _should_retreat(active, bench, {}) == False

    def test_should_retreat_empty_bench(self):
        """Low HP active but empty bench -> should NOT retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=20, max_hp=100)
        assert _should_retreat(active, [], {}) == False

    def test_should_retreat_none_active(self):
        """None active -> should NOT retreat."""
        from tactical_bot import _should_retreat
        assert _should_retreat(None, [make_pokemon(hp=80, max_hp=100)], {}) == False

    def test_should_retreat_all_bench_low_hp(self):
        """Low HP active + all bench also low HP -> should NOT retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=20, max_hp=100)
        bench = [make_pokemon(hp=10, max_hp=100), make_pokemon(hp=15, max_hp=100)]
        assert _should_retreat(active, bench, {}) == False


# ---------------------------------------------------------------------------
# Tests: tactical_bot lethal detection
# ---------------------------------------------------------------------------
class TestTacticalLethal:
    def test_lethal_attack_takes_last_prize(self):
        """Attack that KOs opponent and takes last prize -> SCORE_LETHAL."""
        from tactical_bot import _apply_tactical_overrides, SCORE_LETHAL
        from greedy_bot import _score_option

        op_active = make_pokemon(hp=50, max_hp=100, card_id=1)
        o = make_option(opt_type=OptionType.ATTACK, attack_id=10)
        attack_damage = {10: 100}
        card_table = make_mock_card_table(make_card_data(card_id=1, ex=False))

        base_score = _score_option(o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage)
        result = _apply_tactical_overrides(
            o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage,
            None, op_active, 5, 1, 40, False, False, base_score
        )
        assert result == SCORE_LETHAL

    def test_non_lethal_attack_gets_prize_boost(self):
        """Attack that KOs but doesn't take last prize -> prize boost."""
        from tactical_bot import _apply_tactical_overrides
        from greedy_bot import _score_option

        op_active = make_pokemon(hp=50, max_hp=100, card_id=2)
        o = make_option(opt_type=OptionType.ATTACK, attack_id=10)
        attack_damage = {10: 100}
        card_table = make_mock_card_table(make_card_data(card_id=2, ex=True))

        base_score = _score_option(o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage)
        result = _apply_tactical_overrides(
            o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage,
            None, op_active, 3, 3, 40, False, False, base_score
        )
        assert result > base_score
        assert result == base_score + 2000


# ---------------------------------------------------------------------------
# Tests: tactical_bot ex protection
# ---------------------------------------------------------------------------
class TestTacticalExProtection:
    def test_ex_pokemon_penalized_as_active(self):
        """ex Pokemon gets penalized when selected for active spot."""
        from tactical_bot import _apply_tactical_overrides
        from greedy_bot import _score_option

        ex_poke = make_pokemon(card_id=2, hp=100, max_hp=100)
        o = make_option(opt_type=OptionType.CARD, area=AreaType.BENCH, index=0, player_index=0)
        card_table = make_mock_card_table(make_card_data(card_id=2, ex=True))

        with patch('greedy_bot.get_card', return_value=ex_poke):
            base_score = _score_option(o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {})
        with patch('tactical_bot.get_card', return_value=ex_poke):
            result = _apply_tactical_overrides(
                o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {},
                None, None, 5, 5, 40, False, False, base_score
            )
        assert result < base_score

    def test_non_ex_pokemon_boosted_as_active(self):
        """Non-ex Pokemon gets boosted when selected for active spot."""
        from tactical_bot import _apply_tactical_overrides
        from greedy_bot import _score_option

        normal_poke = make_pokemon(card_id=1, hp=100, max_hp=100)
        o = make_option(opt_type=OptionType.CARD, area=AreaType.BENCH, index=0, player_index=0)
        card_table = make_mock_card_table(make_card_data(card_id=1, ex=False))

        with patch('greedy_bot.get_card', return_value=normal_poke):
            base_score = _score_option(o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {})
        with patch('tactical_bot.get_card', return_value=normal_poke):
            result = _apply_tactical_overrides(
                o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {},
                None, None, 5, 5, 40, False, False, base_score
            )
        assert result > base_score


# ---------------------------------------------------------------------------
# Tests: agent deck selection
# ---------------------------------------------------------------------------
class TestAgentDeckSelection:
    def test_greedy_returns_deck_when_no_select(self):
        """GreedyBot returns deck.csv when obs.select is None."""
        from greedy_bot import agent
        obs_dict = {"select": None, "logs": [], "current": None}
        with patch('greedy_bot.read_deck_csv', return_value=list(range(60))):
            result = agent(obs_dict)
        assert len(result) == 60

    def test_tactical_returns_deck_when_no_select(self):
        """TacticalBot returns deck.csv when obs.select is None."""
        from tactical_bot import agent
        obs_dict = {"select": None, "logs": [], "current": None}
        with patch('tactical_bot.read_deck_csv', return_value=list(range(60))):
            result = agent(obs_dict)
        assert len(result) == 60