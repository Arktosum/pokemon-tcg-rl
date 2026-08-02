import random
from cg.api import to_observation_class, SelectContext, OptionType

class BaseAgentClass:
    def __init__(self, deck_path: str):
        self.deck = []
        with open(deck_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                self.deck.append(int(line))
                if len(self.deck) == 60:
                    break

    def parse_observation(self, obs_dict: dict):
        return to_observation_class(obs_dict)

    def get_random_action(self, obs) -> list[int]:
        select = obs.select
        options = list(range(len(select.option)))
        
        if select.context == SelectContext.SETUP_ACTIVE_POKEMON:
            card_options = [i for i, opt in enumerate(select.option) if opt.type == OptionType.CARD]
            if len(card_options) > 0:
                options = card_options

        count = min(select.minCount, len(options))
        if select.maxCount > 0 and count < select.maxCount:
            count = random.randint(count, min(select.maxCount, len(options)))
        return random.sample(options, count)

    def act(self, obs) -> list[int]:
        return self.get_random_action(obs)

    def __call__(self, obs_dict: dict) -> list[int]:
        obs = self.parse_observation(obs_dict)
        if obs.select is None:
            return self.deck
        return self.act(obs)
