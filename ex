"""
test_bots.py — Rigorous unit tests for rule-based bots.
TITAN V5.0 Rule-Based Bot Curriculum.

Tests cover:
- bot_common: safe_select, hp_ratio, energy_count, prize_value, is_ex, get_card
- greedy_bot: agent deck selection, score_option for all OptionTypes
- tactical_bot: _should_retreat, lethal detection, ex protection

All tests use mock dataclass objects — no live engine required for scoring logic.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
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
    SelectType,
    State,
    PlayerState,
    SpecialConditionType,
)


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------
def make_card(card_id: int = 1, serial: int = 1, player_index: int = 0) -> Card:
    return Card(id=card_id, serial=serial, playerIndex=player_index)


def make_pokemon(
    card_id: int = 1,
    serial: int = 1,
    hp: int = 100,
    max_hp: int = 100,
    appear_this_turn: bool = False,
    energies: list = None,
    energy_cards: list = None,
    tools: list = None,
    pre_evolution: list = None,
) -> Pokemon:
    return Pokemon(
        id=card_id,
        serial=serial,
        hp=hp,
        maxHp=max_hp,
        appearThisTurn=appear_this_turn,
        energies=energies or [],
        energyCards=energy_cards or [],
        tools=tools or [],
        preEvolution=pre_evolution or [],
    )


def make_card_data(
    card_id: int = 1,
    name: str = "TestCard",
    card_type: CardType = CardType.POKEMON,
    ex: bool = False,
    mega_ex: bool = False,
    basic: bool = True,
    stage1: bool = False,
    stage2: bool = False,
    hp: int = 100,
) -> CardData:
    return CardData(
        cardId=card_id,
        name=name,
        cardType=card_type,
        retreatCost=1,
        hp=hp,
        weakness=None,
        resistance=None,
        energyType=EnergyType.COLORLESS,
        basic=basic,
        stage1=stage1,
        stage2=stage2,
        ex=ex,
        megaEx=mega_ex,
        tera=False,
        aceSpec=False,
        evolvesFrom=None,
        skills=[],
        attacks=[],
    )


def make_option(
    opt_type: OptionType = OptionType.END,
    number: Optional[int] = None,
    area: Optional[AreaType] = None,
    index: Optional[int] = None,
    player_index: Optional[int] = None,
    tool_index: Optional[int] = None,
    energy_index: Optional[int] = None,
    count: Optional[int] = None,
    in_play_area: Optional[AreaType] = None,
    in_play_index: Optional[int] = None,
    attack_id: Optional[int] = None,
    card_id: Optional[int] = None,
    serial: Optional[int] = None,
    special_condition_type: Optional[SpecialConditionType] = None,
) -> Option:
    return Option(
        type=opt_type,
        number=number,
        area=area,
        index=index,
        playerIndex=player_index,
        toolIndex=tool_index,
        energyIndex=energy_index,
        count=count,
        inPlayArea=in_play_area,
        inPlayIndex=in_play_index,
        attackId=attack_id,
        cardId=card_id,
        serial=serial,
        specialConditionType=special_condition_type,
    )


def make_select_data(
    options: list[Option] = None,
    context: SelectContext = SelectContext.MAIN,
    min_count: int = 1,
    max_count: int = 1,
    remain_damage_counter: int = 0,
    remain_energy_cost: int = 0,
    deck: Optional[list] = None,
    context_card: Optional[Card] = None,
    effect: Optional[Card] = None,
) -> SelectData:
    return SelectData(
        type=SelectType.MAIN,
        context=context,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=remain_damage_counter,
        remainEnergyCost=remain_energy_cost,
        option=options or [],
        deck=deck,
        contextCard=context_card,
        effect=effect,
    )


def make_player_state(
    active: list = None,
    bench: list = None,
    deck_count: int = 40,
    discard: list = None,
    prize: list = None,
    hand_count: int = 4,
    hand: Optional[list] = None,
    poisoned: bool = False,
    burned: bool = False,
    asleep: bool = False,
    paralyzed: bool = False,
    confused: bool = False,
    bench_max: int = 5,
) -> PlayerState:
    return PlayerState(
        active=active or [],
        bench=bench or [],
        benchMax=bench_max,
        deckCount=deck_count,
        discard=discard or [],
        prize=prize or [],
        handCount=hand_count,
        hand=hand,
        poisoned=poisoned,
        burned=burned,
        asleep=asleep,
        paralyzed=paralyzed,
        confused=confused,
    )


def make_state(
    turn: int = 1,
    your_index: int = 0,
    first_player: int = 0,
    players: list = None,
    active_pokemon: Optional[Pokemon] = None,
    bench_pokemon: list = None,
    hand: Optional[list] = None,
    deck_count: int = 40,
    prize_count: int = 6,
) -> State:
    if players is None:
        my_ps = make_player_state(
            active=[active_pokemon] if active_pokemon else [],
            bench=bench_pokemon or [],
            hand=hand,
            deck_count=deck_count,
            prize=[make_card(i, i, 0) for i in range(prize_count)],
        )
        op_ps = make_player_state(
            active=[make_pokemon(hp=100, max_hp=100)] if active_pokemon else [],
            bench=[],
            prize=[make_card(i, i, 1) for i in range(prize_count)],
        )
        players = [my_ps, op_ps]
    return State(
        turn=turn,
        turnActionCount=0,
        yourIndex=your_index,
        firstPlayer=first_player,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        result=-1,
        stadium=[],
        looking=None,
        players=players,
    )


def make_observation(
    select: Optional[SelectData] = None,
    state: Optional[State] = None,
) -> Observation:
    return Observation(
        select=select,
        logs=[],
        current=state,
    )


def make_mock_card_table(*card_datas: CardData) -> dict[int, CardData]:
    return {cd.cardId: cd for cd in card_datas}


# ---------------------------------------------------------------------------
# Tests: bot_common.safe_select
# ---------------------------------------------------------------------------
class TestSafeSelect:
    """Test the safe_select action selection function."""

    def test_empty_scores_returns_empty(self):
        """safe_select with empty scores returns empty list."""
        select = make_select_data(options=[], min_count=0, max_count=1)
        assert safe_select([], select) == []

    def test_all_negative_min_zero_returns_empty(self):
        """All negative scores with minCount=0 returns empty (skip all)."""
        select = make_select_data(min_count=0, max_count=3)
        assert safe_select([-1.0, -2.0, -3.0], select) == []

    def test_min_count_enforcement(self):
        """minCount=1 with all negative returns 1 item (highest score)."""
        select = make_select_data(min_count=1, max_count=3)
        result = safe_select([-1.0, -2.0, -3.0], select)
        assert len(result) == 1
        assert result[0] == 0  # Index of -1.0 (highest)

    def test_max_count_cap(self):
        """Returns at most maxCount items."""
        select = make_select_data(min_count=0, max_count=2)
        result = safe_select([10.0, 20.0, 30.0], select)
        assert len(result) == 2
        assert result == [2, 1]  # Highest scores first

    def test_ordering_descending(self):
        """Returns indices in descending score order."""
        select = make_select_data(min_count=0, max_count=5)
        result = safe_select([5.0, 3.0, 8.0, 1.0, 6.0], select)
        assert result == [2, 4, 0, 1, 3]

    def test_mixed_scores(self):
        """Mix of positive and negative: only return positive when minCount=0."""
        select = make_select_data(min_count=0, max_count=5)
        result = safe_select([10.0, -5.0, 3.0, -1.0, 0.0], select)
        # 0.0 is not negative, so it should be included
        assert 0 in result  # index of 10.0
        assert 2 in result  # index of 3.0
        assert 4 in result  # index of 0.0
        assert 1 not in result  # -5.0 skipped
        assert 3 not in result  # -1.0 skipped

    def test_no_duplicates(self):
        """No duplicate indices in output."""
        select = make_select_data(min_count=0, max_count=3)
        result = safe_select([1.0, 2.0, 3.0], select)
        assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# Tests: bot_common.hp_ratio
# ---------------------------------------------------------------------------
class TestHpRatio:
    """Test the hp_ratio function."""

    def test_none_returns_zero(self):
        """None Pokemon returns 0.0."""
        assert hp_ratio(None) == 0.0

    def test_zero_max_hp_returns_zero(self):
        """maxHp=0 returns 0.0 (division by zero guard)."""
        poke = make_pokemon(hp=50, max_hp=0)
        assert hp_ratio(poke) == 0.0

    def test_half_hp(self):
        """HP=50, maxHp=100 returns 0.5."""
        poke = make_pokemon(hp=50, max_hp=100)
        assert hp_ratio(poke) == 0.5

    def test_full_hp(self):
        """HP=100, maxHp=100 returns 1.0."""
        poke = make_pokemon(hp=100, max_hp=100)
        assert hp_ratio(poke) == 1.0

    def test_low_hp(self):
        """HP=20, maxHp=100 returns 0.2."""
        poke = make_pokemon(hp=20, max_hp=100)
        assert hp_ratio(poke) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Tests: bot_common.energy_count
# ---------------------------------------------------------------------------
class TestEnergyCount:
    """Test the energy_count function."""

    def test_none_returns_zero(self):
        """None Pokemon returns 0."""
        assert energy_count(None) == 0

    def test_no_energies(self):
        """Pokemon with empty energies returns 0."""
        poke = make_pokemon(energies=[])
        assert energy_count(poke) == 0

    def test_three_energies(self):
        """Pokemon with 3 energies returns 3."""
        poke = make_pokemon(energies=[EnergyType.FIRE, EnergyType.WATER, EnergyType.GRASS])
        assert energy_count(poke) == 3


# ---------------------------------------------------------------------------
# Tests: bot_common.prize_value
# ---------------------------------------------------------------------------
class TestPrizeValue:
    """Test the prize_value function."""

    def test_none_returns_zero(self):
        """None Pokemon returns 0."""
        assert prize_value(None, {}) == 0

    def test_normal_pokemon(self):
        """Normal Pokemon (not ex) returns 1."""
        poke = make_pokemon(card_id=1)
        table = make_mock_card_table(make_card_data(card_id=1, ex=False, mega_ex=False))
        assert prize_value(poke, table) == 1

    def test_ex_pokemon(self):
        """ex Pokemon returns 2."""
        poke = make_pokemon(card_id=2)
        table = make_mock_card_table(make_card_data(card_id=2, ex=True, mega_ex=False))
        assert prize_value(poke, table) == 2

    def test_mega_ex_pokemon(self):
        """megaEx Pokemon returns 3."""
        poke = make_pokemon(card_id=3)
        table = make_mock_card_table(make_card_data(card_id=3, ex=True, mega_ex=True))
        assert prize_value(poke, table) == 3

    def test_legacy_energy_reduces_prize(self):
        """Legacy Energy (id=12) reduces prize count by 1."""
        poke = make_pokemon(card_id=2, energy_cards=[make_card(card_id=12)])
        table = make_mock_card_table(make_card_data(card_id=2, ex=True, mega_ex=False))
        assert prize_value(poke, table) == 1  # 2 - 1 = 1

    def test_prize_never_negative(self):
        """Prize count never goes below 0."""
        poke = make_pokemon(card_id=3, energy_cards=[make_card(card_id=12), make_card(card_id=12), make_card(card_id=12)])
        table = make_mock_card_table(make_card_data(card_id=3, ex=True, mega_ex=True))
        assert prize_value(poke, table) == 0  # 3 - 3 = 0, clamped


# ---------------------------------------------------------------------------
# Tests: bot_common.is_ex
# ---------------------------------------------------------------------------
class TestIsEx:
    """Test the is_ex function."""

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
    """Test GreedyBot's scoring functions."""

    def test_score_attack_with_damage(self):
        """Attack with known damage gets base + damage."""
        from greedy_bot import _score_attack
        attack_damage = {10: 50}
        o = make_option(opt_type=OptionType.ATTACK, attack_id=10)
        score = _score_attack(o, attack_damage)
        assert score == 1000 + 50  # SCORE_ATTACK_BASE + damage

    def test_score_attack_unknown_id(self):
        """Attack with unknown attackId gets base score only."""
        from greedy_bot import _score_attack
        attack_damage = {10: 50}
        o = make_option(opt_type=OptionType.ATTACK, attack_id=999)
        score = _score_attack(o, attack_damage)
        assert score == 1000  # SCORE_ATTACK_BASE only

    def test_score_attack_none_id(self):
        """Attack with None attackId gets base score."""
        from greedy_bot import _score_attack
        o = make_option(opt_type=OptionType.ATTACK, attack_id=None)
        score = _score_attack(o, {})
        assert score == 1000

    def test_score_evolve_high_priority(self):
        """Evolve gets very high score (50000+)."""
        from greedy_bot import _score_evolve, SCORE_EVOLVE
        o = make_option(opt_type=OptionType.EVOLVE, in_play_area=AreaType.ACTIVE, in_play_index=0)
        # Mock get_card to return a Pokemon
        with patch('greedy_bot.get_card', return_value=make_pokemon(energies=[EnergyType.FIRE])):
            score = _score_evolve(o, MagicMock(), 0)
        assert score >= SCORE_EVOLVE

    def test_score_attach_active_zero_energy(self):
        """Attach to active with 0 energy gets highest attach score."""
        from greedy_bot import _score_attach, SCORE_ATTACH
        o = make_option(opt_type=OptionType.ATTACH, in_play_area=AreaType.ACTIVE, in_play_index=0)
        poke = make_pokemon(energies=[])
        with patch('greedy_bot.get_card', return_value=poke):
            score = _score_attach(o, MagicMock(), 0)
        assert score == SCORE_ATTACH + 500

    def test_score_attach_bench_with_energy(self):
        """Attach to benched Pokemon with energy gets base attach score."""
        from greedy_bot import _score_attach, SCORE_ATTACH
        o = make_option(opt_type=OptionType.ATTACH, in_play_area=AreaType.BENCH, in_play_index=0)
        poke = make_pokemon(energies=[EnergyType.FIRE])
        with patch('greedy_bot.get_card', return_value=poke):
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
        score = _score_option(o, MagicMock(), SelectContext.MAIN, 0, {}, {})
        assert score == SCORE_YES

    def test_score_end(self):
        """END option gets score 10."""
        from greedy_bot import _score_option, SCORE_END
        o = make_option(opt_type=OptionType.END)
        score = _score_option(o, MagicMock(), SelectContext.MAIN, 0, {}, {})
        assert score == SCORE_END

    def test_score_retreat_skip(self):
        """RETREAT option gets SCORE_SKIP in GreedyBot."""
        from greedy_bot import _score_option, SCORE_SKIP
        o = make_option(opt_type=OptionType.RETREAT)
        score = _score_option(o, MagicMock(), SelectContext.MAIN, 0, {}, {})
        assert score == SCORE_SKIP


# ---------------------------------------------------------------------------
# Tests: tactical_bot
# ---------------------------------------------------------------------------
class TestTacticalRetreat:
    """Test TacticalBot's retreat logic."""

    def test_should_retreat_low_hp_healthy_bench(self):
        """Low HP active + healthy bench Pokemon → should retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=20, max_hp=100)  # 20% HP
        bench = [make_pokemon(hp=80, max_hp=100)]  # 80% HP
        assert _should_retreat(active, bench, {}) == True

    def test_should_retreat_high_hp(self):
        """High HP active → should NOT retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=90, max_hp=100)  # 90% HP
        bench = [make_pokemon(hp=80, max_hp=100)]
        assert _should_retreat(active, bench, {}) == False

    def test_should_retreat_empty_bench(self):
        """Low HP active but empty bench → should NOT retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=20, max_hp=100)
        assert _should_retreat(active, [], {}) == False

    def test_should_retreat_none_active(self):
        """None active → should NOT retreat."""
        from tactical_bot import _should_retreat
        assert _should_retreat(None, [make_pokemon(hp=80, max_hp=100)], {}) == False

    def test_should_retreat_all_bench_low_hp(self):
        """Low HP active + all bench also low HP → should NOT retreat."""
        from tactical_bot import _should_retreat
        active = make_pokemon(hp=20, max_hp=100)
        bench = [make_pokemon(hp=10, max_hp=100), make_pokemon(hp=15, max_hp=100)]
        assert _should_retreat(active, bench, {}) == False


class TestTacticalLethal:
    """Test TacticalBot's lethal detection."""

    def test_lethal_attack_takes_last_prize(self):
        """Attack that KOs opponent and takes last prize → SCORE_LETHAL."""
        from tactical_bot import _apply_tactical_overrides, SCORE_LETHAL
        from greedy_bot import _score_option

        # Opponent has 1 prize left, attack does 100 damage, opponent has 50 HP
        op_active = make_pokemon(hp=50, max_hp=100, card_id=1)
        o = make_option(opt_type=OptionType.ATTACK, attack_id=10)
        attack_damage = {10: 100}
        card_table = make_mock_card_table(make_card_data(card_id=1, ex=False))

        base_score = _score_option(o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage)

        result = _apply_tactical_overrides(
            o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage,
            None, op_active, 5, 1, 40,  # my_prizes=5, op_prizes=1
            False, False, base_score
        )
        assert result == SCORE_LETHAL

    def test_non_lethal_attack_gets_prize_boost(self):
        """Attack that KOs but doesn't take last prize → prize boost."""
        from tactical_bot import _apply_tactical_overrides
        from greedy_bot import _score_option

        # Opponent has 3 prizes left, attack KOs ex Pokemon (2 prizes)
        op_active = make_pokemon(hp=50, max_hp=100, card_id=2)
        o = make_option(opt_type=OptionType.ATTACK, attack_id=10)
        attack_damage = {10: 100}
        card_table = make_mock_card_table(make_card_data(card_id=2, ex=True))

        base_score = _score_option(o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage)

        result = _apply_tactical_overrides(
            o, MagicMock(), SelectContext.MAIN, 0, card_table, attack_damage,
            None, op_active, 3, 3, 40,  # op_prizes=3, not last
            False, False, base_score
        )
        # Should be base + 2 * 1000 (ex prize boost)
        assert result > base_score
        assert result == base_score + 2000


class TestTacticalExProtection:
    """Test TacticalBot's ex protection logic."""

    def test_ex_pokemon_penalized_as_active(self):
        """ex Pokemon gets penalized when selected for active spot."""
        from tactical_bot import _apply_tactical_overrides
        from greedy_bot import _score_option

        ex_poke = make_pokemon(card_id=2, hp=100, max_hp=100)
        o = make_option(
            opt_type=OptionType.CARD,
            area=AreaType.BENCH,
            index=0,
            player_index=0,
        )
        card_table = make_mock_card_table(make_card_data(card_id=2, ex=True))

        base_score = _score_option(o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {})

        result = _apply_tactical_overrides(
            o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {},
            None, None, 5, 5, 40,
            False, False, base_score
        )
        # ex penalty applied
        assert result < base_score

    def test_non_ex_pokemon_boosted_as_active(self):
        """Non-ex Pokemon gets boosted when selected for active spot."""
        from tactical_bot import _apply_tactical_overrides
        from greedy_bot import _score_option

        normal_poke = make_pokemon(card_id=1, hp=100, max_hp=100)
        o = make_option(
            opt_type=OptionType.CARD,
            area=AreaType.BENCH,
            index=0,
            player_index=0,
        )
        card_table = make_mock_card_table(make_card_data(card_id=1, ex=False))

        base_score = _score_option(o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {})

        result = _apply_tactical_overrides(
            o, MagicMock(), SelectContext.TO_ACTIVE, 0, card_table, {},
            None, None, 5, 5, 40,
            False, False, base_score
        )
        # Non-ex boost applied
        assert result > base_score


# ---------------------------------------------------------------------------
# Tests: agent deck selection
# ---------------------------------------------------------------------------
class TestAgentDeckSelection:
    """Test that agents return deck when obs.select is None."""

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


# ---------------------------------------------------------------------------
# Tests: agent returns valid indices
# ---------------------------------------------------------------------------
class TestAgentValidIndices:
    """Test that agents return valid indices within [0, len(option))."""

    def test_greedy_returns_valid_indices(self):
        """GreedyBot returns indices within valid range."""
        from greedy_bot import agent

        options = [
            make_option(opt_type=OptionType.END),
            make_option(opt_type=OptionType.ATTACK, attack_id=1),
            make_option(opt_type=OptionType.YES),
        ]
        select = make_select_data(options=options, min_count=1, max_count=1)
        state = make_state()
        obs = make_observation(select=select, state=state)
        obs_dict = {
            "select": {
                "type": 0,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "remainDamageCounter": 0,
                "remainEnergyCost": 0,
                "option": [{"type": 14}, {"type": 13, "attackId": 1}, {"type": 1}],
                "deck": None,
                "contextCard": None,
                "effect": None,
            },
            "logs": [],
            "current": {
                "turn": 1,
                "turnActionCount": 0,
                "yourIndex": 0,
                "firstPlayer": 0,
                "supporterPlayed": False,
                "stadiumPlayed": False,
                "energyAttached": False,
                "retreated": False,
                "result": -1,
                "stadium": [],
                "looking": None,
                "players": [
                    {
                        "active": [],
                        "bench": [],
                        "benchMax": 5,
                        "deckCount": 40,
                        "discard": [],
                        "prize": [],
                        "handCount": 4,
                        "hand": [],
                        "poisoned": False,
                        "burned": False,
                        "asleep": False,
                        "paralyzed": False,
                        "confused": False,
                    },
                    {
                        "active": [],
                        "bench": [],
                        "benchMax": 5,
                        "deckCount": 40,
                        "discard": [],
                        "prize": [],
                        "handCount": 4,
                        "hand": None,
                        "poisoned": False,
                        "burned": False,
                        "asleep": False,
                        "paralyzed": False,
                        "confused": False,
                    },
                ],
            },
        }

        with patch('greedy_bot.load_card_table', return_value={}), \
             patch('greedy_bot._load_attack_damage', return_value={1: 50}), \
             patch('greedy_bot.read_deck_csv', return_value=list(range(60))):
            result = agent(obs_dict)

        assert isinstance(result, list)
        assert len(result) >= 1
        for idx in result:
            assert 0 <= idx < len(options)