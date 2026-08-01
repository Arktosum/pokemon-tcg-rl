import torch

class RolloutBuffer:
    def __init__(self):
        self.enc_indices = []
        self.enc_weights = []
        self.enc_offsets = []
        self.dec_indices = []
        self.dec_weights = []
        self.dec_offsets = []
        
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.enc_indices.append(state["enc_indices"])
        self.enc_weights.append(state["enc_weights"])
        self.enc_offsets.append(state["enc_offsets"])
        self.dec_indices.append(state["dec_indices"])
        self.dec_weights.append(state["dec_weights"])
        self.dec_offsets.append(state["dec_offsets"])
        
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def clear(self):
        self.enc_indices.clear()
        self.enc_weights.clear()
        self.enc_offsets.clear()
        self.dec_indices.clear()
        self.dec_weights.clear()
        self.dec_offsets.clear()
        
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        
    def compute_returns_and_advantages(self, last_value, done, gamma=0.99, gae_lambda=0.95):
        returns = []
        advantages = []
        
        gae = 0
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0 - float(done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[step + 1])
                next_value = self.values[step + 1]
                
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
            
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)
