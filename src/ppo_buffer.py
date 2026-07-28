import numpy as np
import torch

class PPOBuffer:
    def __init__(self, gamma=0.99, lam=0.95):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.action_masks = []
        self.gamma = gamma
        self.lam = lam
        
    def store(self, state, action_mask, action, reward, value, log_prob, mask):
        self.states.append(state)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.action_masks.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()
        
    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae = 0
        returns = []
        advantages = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + self.gamma * self.lam * self.masks[step] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return advantages, returns
