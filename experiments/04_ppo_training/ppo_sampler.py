# In ppo_sampler.py
import glob
import os
import random
from ppo_bridge import generate_selfplay_script


class OpponentSampler:

    def __init__(self, curr_cfg):
        self.cfg = curr_cfg
        self.checkpoints_dir = os.path.join(
            os.path.dirname(__file__), "checkpoints"
        )
        self.agents_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "03_rule_based_eval", "agents"
            )
        )

    def _get_available_rule_bots(self):
        return self.cfg.rule_based_bots

    def sample_opponent(self):
        latest_exists = os.path.exists(
            os.path.join(self.checkpoints_dir, "latest.pt")
        )
        best_exists = os.path.exists(
            os.path.join(self.checkpoints_dir, "best.pt")
        )

        choices = []
        weights = []

        if latest_exists:
            choices.append("latest_checkpoint")
            weights.append(
                self.cfg.sampling_weights.get("latest_checkpoint", 0.3)
            )

        if best_exists:
            choices.append("best_checkpoint")
            weights.append(self.cfg.sampling_weights.get("best_checkpoint", 0.1))

        choices.append("random_action")
        weights.append(self.cfg.sampling_weights.get("random_action", 0.1))

        choices.append("rule_based")
        weights.append(self.cfg.sampling_weights.get("rule_based", 0.5))

        total = sum(weights)
        weights = [w / total for w in weights]

        selected = random.choices(choices, weights=weights, k=1)[0]

        if selected == "latest_checkpoint":
            return generate_selfplay_script("latest.pt"), "Self-Play (Latest)"
        elif selected == "best_checkpoint":
            return generate_selfplay_script("best.pt"), "Self-Play (Best)"
        elif selected == "random_action":
            return "random", "Random Bot"
        else:
            available_bots = self._get_available_rule_bots()
            bot_file = random.choice(available_bots)
            bot_path = os.path.join(self.agents_dir, bot_file)
            display_name = bot_file.replace(".py", "").replace("_fixed", "")
            return bot_path, display_name