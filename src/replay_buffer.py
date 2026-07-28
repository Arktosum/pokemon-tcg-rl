import collections
import numpy as np
import torch
import gc

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        
    def save_trajectory(self, trajectory, winner):
        for step in trajectory:
            # Perspective inversion Bug Prevention: +1 if this player won, -1 if they lost, 0 if draw
            if winner == -1:
                z = 0.0
            else:
                z = 1.0 if step['player'] == winner else -1.0
                
            # Explicit CPU Storage to prevent RAM/VRAM leak
            self.buffer.append({
                'state': np.copy(step['state']),
                'mask': np.copy(step['mask']),
                'policy': np.copy(step['policy']),
                'z': np.float32(z)
            })
            
        # VRAM/RAM Leakage prevention
        del trajectory
        gc.collect()
        
    def sample_batch(self, batch_size):
        replace = len(self.buffer) < batch_size
        indices = np.random.choice(len(self.buffer), batch_size, replace=replace)
        states, masks, policies, zs = [], [], [], []
        
        for idx in indices:
            item = self.buffer[idx]
            states.append(item['state'])
            masks.append(item['mask'])
            policies.append(item['policy'])
            zs.append(item['z'])
            
        return {
            'state': torch.tensor(np.array(states), dtype=torch.float32),
            'mask': torch.tensor(np.array(masks), dtype=torch.int8),
            'policy': torch.tensor(np.array(policies), dtype=torch.float32),
            'z': torch.tensor(np.array(zs), dtype=torch.float32).unsqueeze(1)
        }
    
    def __len__(self):
        return len(self.buffer)
