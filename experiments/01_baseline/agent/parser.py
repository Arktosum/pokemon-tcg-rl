import sys
import os
import json
import torch
import torch.nn as nn

from agent.cg.api import (
    to_observation_class,
    all_card_data,
    Observation,
    PlayerState,
    Pokemon,
    Card,
    all_attack,
    OptionType,
    SelectContext,
    AreaType
)

# Fetch total unique cards dynamically
all_card = all_card_data()
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1
attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1

# Dynamically compute the maximum possible index (encoder_size)
# 12 pokemon slots * (1 + 3 * card_count)
# + 2 player slots * 7
# + 1 hand * card_count
# + 1 deck * card_count (approximate buffer)
encoder_size = 12 * (1 + 3 * card_count) + 2 * 7 + 2 * card_count + 1000

# Action decoder uses the same namespace
decoder_size = encoder_size

# Decoder constants
decoder_main_feature = 8 # Feature count of SelectContext.Main
decoder_attack_offset = 14 # First index of Attack feature
decoder_card_offset = decoder_attack_offset + attack_count # First index of Card Feature
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count

class SparseVector:
    """Helper class to aggregate ids, weights, and offsets for EmbeddingBag"""
    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        """Marks the start of a new 'word' (container slot)"""
        self.offset.append(len(self.index))


def add_card(sv: SparseVector, card: Card | Pokemon | None):
    if card is not None:
        sv.add(card.id, 1.0)
    sv.add_pos(card_count)

def add_cards(sv: SparseVector, cards: list[Card] | None, value: float):
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)

def add_pokemon(sv: SparseVector, poke: Pokemon | None):
    if poke is None:
        sv.add_single(1) # Flag indicating empty slot
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0) # Flag indicating occupied slot
        sv.add_single(poke.hp / 400.0) # HP Ratio
        add_card(sv, poke) # Pokemon ID
        add_cards(sv, poke.tools, 1.0) # Tools
        add_cards(sv, poke.energyCards, 0.5) # Energies
        
def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60.0)
    sv.add_single(len(ps.discard) / 60.0)
    sv.add_single(ps.handCount / 8.0)
    sv.add_single(len(ps.bench) / 5.0)
    sv.add(sum(1 for p in ps.prize if p is not None), 1.0)
    sv.add_pos(7) # Shift by max prize + flags

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs: Observation, your_deck: list[int] = None) -> SparseVector:
    if your_deck is None:
        your_deck = []
    """Parses the entire observation into exactly 24 PyTorch words."""
    your_index = obs.current.yourIndex
    state = obs.current
    sv = SparseVector()
    
    # 1. Benches (16 Words: 8 Opponent, 8 Ours)
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos # Reset namespace for next bench slot
    
    # 2. Actives (2 Words: Opponent, Ours)
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if len(ps.active) > 0:
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    # 3. Player States (2 Words: Opponent, Ours)
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)
        
    # 4. Our Hand (1 Word)
    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)
        
    # 5. Our Deck (1 Word)
    sv.word_start()
    for card_id in your_deck:
        sv.add(card_id, 0.25)
    sv.add_pos(card_count)
        
    # 6. Stadium (1 Word)
    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    # 7. Global State (1 Word)
    sv.word_start()
    sv.add_single(1) # Bias
    sv.add_single(state.turn / 10.0)
    sv.add_single(state.firstPlayer == your_index)
    
    return sv


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK: return obs.select.deck[index]
        case AreaType.HAND: return ps.hand[index]
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE: return ps.active[index]
        case AreaType.BENCH: return ps.bench[index]
        case AreaType.PRIZE: return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index]
        case _: return None

def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None):
    if card is not None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)

def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)

def decoder_card(sv: SparseVector, context: SelectContext, card: Card | Pokemon | None):
    if card is not None:
        decoder_card_id(sv, context, card.id)

def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    """Parses a list of legal actions into PyTorch bags."""
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    
    for action in actions:
        sv.word_start()
        
        if len(action) == 0:
            sv.add(0, 1) # Null action
            continue
            
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END: sv.add(1, 1)
                case OptionType.YES: sv.add(2, 1)
                case OptionType.NO: sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION: sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER: sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK: sv.add(decoder_attack_offset + o.attackId, 1)
                case OptionType.PLAY: decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY: decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD: decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT: decoder_main(sv, 7, ps.active[0] if len(ps.active) > 0 else None)
                case OptionType.CARD: decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    if card is not None and len(card.tools) > o.toolIndex:
                        decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    if card is not None and len(card.energyCards) > o.energyIndex:
                        decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL: decoder_card_id(sv, context, o.cardId)
    return sv


if __name__ == '__main__':
    print("Loading replay_20260730_125951.json...")
    replay_path = os.path.join(os.path.dirname(__file__), '..', 'replay_20260730_125951.json')
    with open(replay_path, 'r', encoding='utf-8') as f:
        replay = json.load(f)
        
    # Extract a mid-game step (e.g. step 10) to ensure board is populated
    step_data = replay['steps'][10]
    # step_data has 2 elements (Player 0 and Player 1)
    obs_dict = step_data[0]['observation']
    
    # Parse into Python classes
    obs = to_observation_class(obs_dict)
    
    # Run the Sparse parser
    print("Parsing observation into SparseVector...")
    sv = get_encoder_input(obs)
    
    print(f"Total tokens collected: {len(sv.index)}")
    print(f"Total words (slots) created: {len(sv.offset)}")
    
    assert len(sv.offset) == 24, "Parser failed to output exactly 24 words!"
    
    # Create the EmbeddingBag to prove the tensor transformation
    embedding_dim = 128
    # Use sv.pos (the total vocabulary namespace size) as num_embeddings
    bag = nn.EmbeddingBag(num_embeddings=sv.pos, embedding_dim=embedding_dim, mode='sum')
    
    # Convert to tensors
    indices = torch.tensor(sv.index, dtype=torch.int32)
    values = torch.tensor(sv.value, dtype=torch.float32)
    offsets = torch.tensor(sv.offset, dtype=torch.int32)
    
    # We pass the bag the values as per-sample weights!
    output = bag(indices, offsets, per_sample_weights=values)
    
    print("\n--- ENCODER TENSOR PROOF ---")
    print(f"Output Shape: {output.shape}")
    print("Shape Verification: [24 Words, 128 Dimensions]")
    print("The 24 containers have been successfully populated!")
    
    # Run the Decoder parser
    print("\nParsing legal actions into Decoder SparseVector...")
    
    # Mock some action sequences based on available options
    # We will just evaluate each option as a single distinct action
    legal_option_count = len(obs.select.option)
    actions = [[i] for i in range(legal_option_count)]
    
    sv_dec = get_decoder_input(obs, actions)
    
    print(f"Legal Moves Found: {legal_option_count}")
    print(f"Total tokens collected for actions: {len(sv_dec.index)}")
    print(f"Total action words created: {len(sv_dec.offset)}")
    
    # Create the Decoder EmbeddingBag
    bag_dec = nn.EmbeddingBag(num_embeddings=decoder_size, embedding_dim=embedding_dim, mode='sum')
    
    dec_indices = torch.tensor(sv_dec.index, dtype=torch.int32)
    dec_values = torch.tensor(sv_dec.value, dtype=torch.float32)
    dec_offsets = torch.tensor(sv_dec.offset, dtype=torch.int32)
    
    output_dec = bag_dec(dec_indices, dec_offsets, per_sample_weights=dec_values)
    
    print("\n--- DECODER TENSOR PROOF ---")
    print(f"Output Shape: {output_dec.shape}")
    print(f"Shape Verification: [{legal_option_count} Legal Actions, 128 Dimensions]")
    print("The Decoder action words have been successfully generated!")
