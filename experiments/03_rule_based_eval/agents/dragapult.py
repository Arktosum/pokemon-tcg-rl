import os
import sys

if '__file__' in globals():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
else:
    base_dir = os.getcwd()

baseline_agent_dir = os.path.abspath(os.path.join(base_dir, "experiments", "01_baseline", "agent"))
if baseline_agent_dir not in sys.path:
    sys.path.insert(0, baseline_agent_dir)

from collections import defaultdict
from cg.api import AreaType, CardType, Log, LogType, Observation, SelectContext, OptionType, Card, Pokemon, State, all_card_data, to_observation_class
from base_agent import BaseAgentClass

all_card = all_card_data()
card_table = {c.cardId:c for c in all_card}

Dreepy = 119
Drakloak = 120
Dragapult_ex = 121
Fezandipiti_ex = 140
Latias_ex = 184
Budew = 235
Meowth_ex = 1071
Rare_Candy = 1079
Unfair_Stamp = 1080
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Crushing_Hammer = 1120
Ultra_Ball = 1121
Poke_Pad = 1152
Lucky_Helmet = 1156
Boss_Orders = 1182
Crispin = 1198
Brock_Scouting = 1210
Lillie_Determination = 1227
Team_Rocket_Watchtower = 1256
Basic_Fire_Energy = 2
Basic_Psychic_Energy = 5

UNNECESSARY = -10000000

class AttackPlan:
    def __init__(self):
        self.attack: int = 0
        self.counter: list[int] = []

class DragapultAgent(BaseAgentClass):
    def __init__(self, deck_path: str):
        super().__init__(deck_path)
        self.can_switch = False
        self.can_attack = False
        self.can_main_attack = False
        self.can_energy_attach = False
        self.use_support = 0
        self.bench_attacker = False
        self.pre_turn_log = []
        self.current_turn_log = []
        self.prize = []
        self.card_counts = defaultdict(int)
        self.serial_set = set()
        self.plan_a = AttackPlan()
        self.plan_b = AttackPlan()

    def no_damage_dex(self, id: int) -> bool:
        return id in (158, 207, 330, 345)

    def no_damage_counter(self, pokemon: Pokemon) -> bool:
        if pokemon.id in (28, 199, 203, 207, 362, 1136):
            return True
        for card in pokemon.energyCards:
            if card.id in (11, 20):
                return True
        return False

    def prize_count(self, pokemon: Pokemon, is_attack_damage: bool) -> int:
        data = card_table[pokemon.id]
        count = 3 if data.megaEx else 2 if data.ex else 1
        if is_attack_damage:
            for card in pokemon.energyCards:
                if card.id == 12:
                    count -= 1
            for card in pokemon.tools:
                if card.id == 1172 and 'Lillie' in data.name:
                    count -= 1
        return max(0, count)

    def pokemon_score(self, pokemon: Pokemon, is_attack_damage: bool) -> int:
        data = card_table[pokemon.id]
        score = self.prize_count(pokemon, is_attack_damage) * 1000
        score += len(pokemon.energies) * 150
        score += len(pokemon.tools) * 100
        if data.stage2:
            score += 250
        elif data.stage1:
            score += 130
        id = pokemon.id
        if id in (173, 174, 190, 1071):
            score -= 200
        if id == 112 and len(pokemon.energies) >= 1:
            score += 300
        score += pokemon.hp
        return score

    def add_card_count(self, card: Card | Pokemon | None, my_index: int):
        if card is None:
            return
        if isinstance(card, Pokemon) or card.playerIndex == my_index:
            if card.serial not in self.serial_set:
                self.card_counts[card.id] -= 1
                self.serial_set.add(card.serial)
        if isinstance(card, Pokemon):
            for c in card.energyCards:
                self.add_card_count(c, my_index)
            for c in card.tools:
                self.add_card_count(c, my_index)
            for c in card.preEvolution:
                self.add_card_count(c, my_index)

    def set_card_counts(self, obs: Observation, my_index: int):
        self.card_counts.clear()
        self.serial_set.clear()
        for id in self.deck:
            self.card_counts[id] += 1
        state = obs.current
        my_state = state.players[my_index]
        for card in my_state.hand:
            self.add_card_count(card, my_index)
        for card in my_state.discard:
            self.add_card_count(card, my_index)
        for card in my_state.bench:
            self.add_card_count(card, my_index)
        for card in my_state.active:
            self.add_card_count(card, my_index)
        for card in state.stadium:
            self.add_card_count(card, my_index)
        if state.looking is not None:
            for card in state.looking:
                self.add_card_count(card, my_index)
        self.add_card_count(obs.select.effect, my_index)

    def get_card(self, obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
        ps = obs.current.players[player_index]
        match area:
            case AreaType.DECK:
                return obs.select.deck[index]
            case AreaType.HAND:
                return ps.hand[index]
            case AreaType.DISCARD:
                return ps.discard[index]
            case AreaType.ACTIVE:
                return ps.active[index]
            case AreaType.BENCH:
                return ps.bench[index]
            case AreaType.PRIZE:
                return ps.prize[index]
            case AreaType.STADIUM:
                return obs.current.stadium[index]
            case AreaType.LOOKING:
                return obs.current.looking[index]
            case _:
                return None

    def main_option_proc(self, obs: Observation, damage: int):
        state = obs.current
        select = obs.select
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]

        self.can_switch = False
        self.can_attack = False
        self.can_main_attack = False
        self.can_energy_attach = False
        for o in select.option:
            if o.type == OptionType.RETREAT:
                self.can_switch = True
            elif o.type == OptionType.ATTACK:
                self.can_attack = True
                if o.attackId == 154:
                    self.can_main_attack = True
        
        self.plan_a.attack = -1
        self.plan_b.attack = -1
        if not self.can_main_attack and not (self.bench_attacker and self.can_switch):
            return
        
        cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            cards.append(pokemon)
        counter_indices = []
        ci = []
        ci.append(0)
        remain_damage = 60
        while ci:
            index = ci[-1]
            hp = cards[index].hp
            if remain_damage >= hp:
                counter_indices.append(ci.copy())
                if index < len(cards) - 1:
                    remain_damage -= hp
                    ci.append(index + 1)
                    continue
            if index == len(cards) - 1:
                ci.pop()
                if ci:
                    remain_damage += cards[ci[-1]].hp
            if ci:
                ci[-1] += 1
        counter_indices.append([])

        remain_prize = len(my_state.prize)
        plan_score = 0
        for i, pokemon in enumerate(cards):
            base_prize_count = 0
            base_score = self.pokemon_score(pokemon, True)
            active_damage = 0 if self.no_damage_dex(pokemon.id) else damage
            if pokemon.hp <= active_damage:
                base_prize_count += self.prize_count(pokemon, True)
            else:
                base_score *= active_damage / pokemon.hp
            ci = []
            max_score = base_score
            if remain_prize <= base_prize_count:
                max_score = 50000
            else:
                for indices in counter_indices:
                    if i in indices:
                        continue
                    prize = base_prize_count
                    score = base_score
                    for index in indices:
                        prize += self.prize_count(cards[index], False)
                        score += self.pokemon_score(cards[index], False)
                    if remain_prize <= prize:
                        score = 50000
                    else:
                        if prize >= 2:
                            if remain_prize <= 4:
                                score -= 1200
                        elif prize == 1:
                            score -= 300
                        else:
                            score += 1200
                    if max_score < score:
                        max_score = score
                        ci = indices
            if plan_score < max_score:
                plan_score = max_score
                self.plan_a.attack = i
                self.plan_a.counter = ci
            if i == 0:
                self.plan_b.attack = self.plan_a.attack
                self.plan_b.counter = self.plan_a.counter

    def act(self, obs: Observation) -> list[int]:
        state = obs.current
        select = obs.select
        context = select.context
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]
                
        if state.turn == 0:
            self.prize.clear()
            self.pre_turn_log.clear()
            self.current_turn_log.clear()
        else:
            for log in obs.logs:
                self.current_turn_log.append(log)
                if log.type == LogType.TURN_END:
                    self.pre_turn_log = self.current_turn_log
                    self.current_turn_log = []

        pre_ko = False
        no_item = False
        for log in self.pre_turn_log:
            if log.type == LogType.ATTACK:
                if log.attackId == 323:
                    no_item = True
            elif log.type == LogType.MOVE_CARD:
                if (log.playerIndex == my_index
                    and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                    and log.toArea == AreaType.DISCARD):
                    pre_ko = True

        if select.deck is not None:
            self.set_card_counts(obs, my_index)
            for card in select.deck:
                self.card_counts[card.id] -= 1
            self.prize.clear()
            for id in self.card_counts:
                for _ in range(self.card_counts[id]):
                    self.prize.append(id)
                    
        self.set_card_counts(obs, my_index)
        for id in self.prize:
            self.card_counts[id] -= 1
        deck_counts = self.card_counts

        prize_diff = len(my_state.prize) - len(op_state.prize)
        
        field_counts = defaultdict(int)
        hand_counts = defaultdict(int)
        discard_counts = defaultdict(int)
        
        active_id = 0
        self.bench_attacker = False
        can_evolve_dreepy = False
        evolve_dreepy_count = 0
        can_evolve_drakloak = False
        damage = 200
        for card in my_state.active:
            if card is None:
                continue
            active_id = card.id
            field_counts[card.id] += 1
            if not card.appearThisTurn:
                if card.id == Dreepy:
                    can_evolve_dreepy = True
                    evolve_dreepy_count += 1
                elif card.id == Drakloak:
                    can_evolve_drakloak = True
        for card in my_state.bench:
            field_counts[card.id] += 1
            if not card.appearThisTurn:
                if card.id == Dreepy:
                    can_evolve_dreepy = True
                    evolve_dreepy_count += 1
                elif card.id == Drakloak:
                    can_evolve_drakloak = True
            if card.id == Dragapult_ex and len(card.energies) >= 2:
                self.bench_attacker = True
        main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
        no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

        stadium_id = 0
        for card in state.stadium:
            stadium_id = card.id

        support_count = 0

        for card in my_state.discard:
            discard_counts[card.id] += 1

        def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
            energy_count = len(pokemon.energies)
            if card_table[attach_id].cardType == CardType.TOOL:
                score = 60000
                if active:
                    score += 1000
                return score
            
            if pokemon.id == Budew:
                return -1
            elif pokemon.id == Meowth_ex or pokemon.id == Fezandipiti_ex or pokemon.id == Latias_ex:
                if active and not self.can_switch and not my_state.asleep and not my_state.paralyzed:
                    if self.bench_attacker or field_counts[Budew] >= 1:
                        return 22000
                    else:
                        return 18000
                else:
                    return -1
            if active and self.can_main_attack:
                return -1
            score = 20000
            if energy_count >= 2:
                if active and not self.can_switch and not my_state.asleep and not my_state.paralyzed:
                    score += 200
                else:
                    return -1
            elif energy_count == 1:
                if attach_id == pokemon.energyCards[0].id:
                    return -1
                if pokemon.id == Dragapult_ex:
                    score += 250
                elif pokemon.id == Dreepy:
                    score -= 150
                else:
                    score -= 200
                if active:
                    score += 200
            else:
                if active:
                    if self.bench_attacker:
                        score += 400
                else:
                    if pokemon.id == Dragapult_ex:
                        score += 150
                    elif pokemon.id == Dreepy:
                        score += 100
                    else:
                        score += 50
                    if self.bench_attacker:
                        score -= 200
            if no_more_dex and (pokemon.id == Dreepy or pokemon.id == Drakloak):
                score -= 500
            return score
        
        def hand_score(id: int, ignore_count: bool):
            score = 0
            if id == Dreepy:
                if main_pokemon_count >= 3:
                    score = 1000
                else:
                    score = 18000
            elif id == Drakloak:
                if can_evolve_dreepy:
                    score = 20000
                else:
                    score = 3000
            elif id == Dragapult_ex:
                if no_more_dex:
                    score = UNNECESSARY
                elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                    score = 40000
                elif can_evolve_drakloak:
                    if field_counts[id] == 0:
                        score = 30000
                    elif field_counts[id] == 1:
                        score = 10000
                    else:
                        score = 50
                else:
                    if field_counts[id] >= 2:
                        score = 50
                    else:
                        score = 2000
            elif id == Fezandipiti_ex:
                if pre_ko:
                    score = 50000
                elif prize_diff <= -2:
                    score = 5
                elif len(op_state.prize) == 1:
                    score = UNNECESSARY
            elif id == Latias_ex:
                if active_id == Fezandipiti_ex or active_id == Meowth_ex or active_id == Dreepy:
                    if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                        score = 28000
                    else:
                        score = 15000
                else:
                    score = 10
            elif id == Budew:
                if field_counts[id] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                    score = UNNECESSARY
                elif state.turn >= 2:
                    score = 30000
            elif id == Meowth_ex:
                if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                    score = 5
                elif state.supporterPlayed:
                    score = 40
                else:
                    score = 35000
            elif id == Rare_Candy:
                if no_more_dex:
                    score = UNNECESSARY
                elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1:
                    score = 40000
            elif id == Unfair_Stamp:
                if pre_ko:
                    score = 80000
                elif len(op_state.prize) == 1:
                    score = UNNECESSARY
                else:
                    score = 80
            elif id == Buddy_Buddy_Poffin:
                count = deck_counts[Dreepy]
                if count == 0:
                    score = UNNECESSARY
                else:
                    if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                        count += 1
                    if count >= 2:
                        score = 35000
            elif id == Night_Stretcher:
                for i in discard_counts:
                    if discard_counts[i] >= 1:
                        card_type = card_table[i].cardType
                        if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                            score = max(score, hand_score(i, ignore_count))
            elif id == Crushing_Hammer:
                score = 20
            elif id == Ultra_Ball:
                if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                    score = 70
                else:
                    score = 5
            elif id == Poke_Pad:
                score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
            elif id == Lucky_Helmet:
                score = 15
            elif id == Boss_Orders:
                if self.plan_a.attack > 0:
                    score = 60000
            elif id == Crispin:
                if not ignore_count or support_count == 0:
                    if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                        score = 10
                    if not self.can_main_attack and not self.bench_attacker and field_counts[Dragapult_ex] >= 1:
                        score = 55000
                    else:
                        score = 25000
            elif id == Brock_Scouting:
                if not ignore_count or support_count == 0:
                    if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                        score = 50000
                    else:
                        score = 30000
            elif id == Lillie_Determination:
                if not ignore_count or support_count == 0:
                    score = 45000
            elif id == Team_Rocket_Watchtower:
                if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                    score = 4000
            elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy:
                if self.can_main_attack and (len(op_state.prize) <= 2
                    or (self.bench_attacker and len(op_state.prize) <= 4)):
                    score = UNNECESSARY
                else:
                    max_score = -10000
                    for pokemon in my_state.active:
                        if pokemon is None:
                            continue
                        max_score = max(max_score, attach_score(id, pokemon, True))
                    for pokemon in my_state.bench:
                        max_score = max(max_score, attach_score(id, pokemon, False))
                    score = max_score - 5000
                    if self.can_main_attack or self.bench_attacker:
                        score /= 10
            
            if not ignore_count and hand_counts[id] > 0:
                if id == Drakloak and hand_counts[id] < evolve_dreepy_count:
                    score -= 10
                elif id == Dreepy:
                    score -= 100
                else:
                    score -= 100000
            return score

        if context == SelectContext.MAIN:
            self.main_option_proc(obs, damage)
                        
            self.use_support = 0
            if not state.supporterPlayed:
                support_score = 0
                for o in select.option:
                    if o.type == OptionType.PLAY:
                        card = self.get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                        if card_table[card.id].cardType == CardType.SUPPORTER:
                            score = hand_score(card.id, True)
                            if support_score < score:
                                support_score = score
                                self.use_support = card.id

        hand_scores = []
        negative_hand_count = 0
        for card in my_state.hand:
            score = hand_score(card.id, False)
            hand_scores.append(score)
            if score < 0:
                negative_hand_count += 1
            hand_counts[card.id] += 1
            if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
                support_count += 1

        no_draw = (my_state.deckCount <= 8)
        do_switch = (not self.can_main_attack and (self.bench_attacker or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
        effect_card_id = 0 if select.effect is None else select.effect.id
        context_card_id = 0 if select.contextCard is None else select.contextCard.id
        
        scores = []
        for o in select.option:
            score = 0
            if o.type == OptionType.NUMBER:
                score = o.number
            elif o.type == OptionType.YES:
                if context == SelectContext.IS_FIRST:
                    score = -1
                else:
                    score = 1
            elif o.type == OptionType.CARD:
                card = self.get_card(obs, o.area, o.index, o.playerIndex)
                if card is not None:
                    energy_count = 0
                    hp = 0
                    if isinstance(card, Pokemon):
                        energy_count = len(card.energies)
                        hp = card.hp
                    if (context == SelectContext.SWITCH
                        or context == SelectContext.TO_ACTIVE
                        or context == SelectContext.SETUP_ACTIVE_POKEMON):
                        if o.playerIndex == my_index:
                            if card.id == Dreepy:
                                score += 10000
                            elif card.id == Drakloak:
                                if energy_count >= 1:
                                    score += 20000
                                else:
                                    score -= 10000
                            elif card.id == Dragapult_ex:
                                score += 50000
                            elif card.id == Budew:
                                if context != SelectContext.SWITCH:
                                    score += 100000
                                elif not self.bench_attacker:
                                    score += 30000
                            elif card.id == Fezandipiti_ex:
                                score -= 1000
                            elif card.id == Meowth_ex:
                                score -= 2000
                        else:
                            if self.plan_a.attack == o.index + 1:
                                score += 100000
                        score += energy_count * 1000
                        score += hp
                    elif context == SelectContext.SETUP_BENCH_POKEMON:
                        if my_index == state.firstPlayer or card.id != Dreepy:
                            score = -1
                    elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                        score = hand_score(card.id, False)
                        hand_counts[card.id] += 1
                        if effect_card_id == Crispin:
                            score = 100000 - hand_score(card.id, True)
                    elif context == SelectContext.DISCARD:
                        hand_counts[card.id] -= 1
                        if card_table[card.id].cardType == CardType.SUPPORTER:
                            support_count -= 1
                        score = -hand_score(card.id, False)
                    elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                        if hp > 0:
                            score = 100000 - 10 * hp + self.pokemon_score(card, False)
                            if context == SelectContext.DAMAGE_COUNTER:
                                if 210 <= hp <= 230:
                                    score += 20000 + hp * 20
                                    if o.area == AreaType.ACTIVE:
                                        score += 10000
                                elif 40 <= hp <= 90:
                                    score += 10000 + hp * 20
                                elif hp <= 30:
                                    score += -10000 + hp * 20
                                if card.id == 133 or card.id == 351:
                                    score += 30000
                            else:
                                index = o.index + 1
                                if index in self.plan_b.counter:
                                    score += 100000
                                else:
                                    remain_damage = select.remainDamageCounter * 10
                                    if 210 <= hp <= 200 + remain_damage:
                                        score += 30000
                                    elif 20 <= hp <= 60 + remain_damage:
                                        score += 10000
                                    elif hp == 10:
                                        score -= 100000
                                if self.no_damage_counter(card):
                                    score = -1
                    elif context == SelectContext.ATTACH_FROM:
                        score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                        if card.id == Dragapult_ex:
                            score += 200
            elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
                if o.playerIndex != state.yourIndex:
                    if o.area == AreaType.BENCH:
                        score = 20
                    else:
                        score = 10
                    card = self.get_card(obs, o.area, o.index, o.playerIndex)
                    if card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                        score += 1
            elif o.type == OptionType.PLAY:
                card = self.get_card(obs, AreaType.HAND, o.index, my_index)
                card_score = hand_scores[o.index]
                if card.id == Dreepy:
                    score = 51000
                elif card.id == Fezandipiti_ex:
                    if card_score > 0:
                        score = 53000
                    else:
                        score = -1
                elif card.id == Latias_ex:
                    if active_id != Drakloak and active_id != Dragapult_ex:
                        score = 51000
                    else:
                        score = -1
                elif card.id == Budew:
                    if field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0:
                        score = 52000
                    else:
                        score = -1
                elif card.id == Meowth_ex:
                    if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                        score = -1
                    elif support_count == 0:
                        score = 50000
                    elif support_count == hand_counts[Boss_Orders] and not self.plan_a.attack <= 0:
                        score = 50000
                    else:
                        score = -1
                elif card.id == Rare_Candy:
                    if no_more_dex:
                        score = -1
                    else:
                        score = 75000
                elif card.id == Unfair_Stamp:
                    score = 15000
                elif card.id == Night_Stretcher:
                    if card_score >= 18000:
                        score = 42000
                    else:
                        score = -1
                elif card.id == Crushing_Hammer:
                    score = 40000
                elif card.id == Boss_Orders:
                    if card.id == self.use_support:
                        score = 35000
                    else:
                        score = -1
                elif card.id == Lillie_Determination:
                    if card.id == self.use_support:
                        score = 14000
                    else:
                        score = -1
                elif card.id == Team_Rocket_Watchtower:
                    if stadium_id > 0 or state.turn == 1:
                        score = 80000
                    else:
                        score = -1
                elif no_draw:
                    score = -1
                elif card.id == Buddy_Buddy_Poffin:
                    if deck_counts[Dreepy] > 0:
                        score = 46000
                    else:
                        score = -1
                elif card.id == Ultra_Ball:
                    if negative_hand_count >= 2:
                        score = 44000
                    else:
                        score = -1
                elif card.id == Poke_Pad:
                    if deck_counts[Dreepy] + deck_counts[Drakloak] > 0:
                        score = 45000
                    else:
                        score = -1
                elif card.id == Crispin or card.id == Brock_Scouting:
                    if card.id == self.use_support:
                        score = 35000
                    else:
                        score = -1
            elif o.type == OptionType.ATTACH:
                card = self.get_card(obs, o.area, o.index, my_index)
                pokemon = self.get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
            elif o.type == OptionType.EVOLVE:
                pokemon = self.get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                score += len(pokemon.energies)
                if pokemon.id == Dreepy:
                    score += 30000
                elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                    score = -1
                else:
                    score += 70000
            elif o.type == OptionType.ABILITY:
                card = self.get_card(obs, o.area, o.index, my_index)
                if no_draw:
                    score = -1
                elif card.id == 1267:
                    score = 1
                else:
                    score = 40000
            elif o.type == OptionType.RETREAT:
                if do_switch:
                    score = 10000
                else:
                    score = -1
            elif o.type == OptionType.ATTACK:
                score = o.attackId

            scores.append(score)

        output = []
        if len(scores) >= 1:
            sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            for i in range(select.maxCount):
                if (sorted_scores[i][1] >= 0
                    or select.minCount > i
                    or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                    output.append(sorted_scores[i][0])
        
        if len(output) >= select.minCount:
            return output
        return super().act(obs)

deck_path = os.path.join(baseline_agent_dir, "deck.csv")
_dragapult_instance = DragapultAgent(deck_path=deck_path)
def agent(obs_dict):
    return _dragapult_instance(obs_dict)
