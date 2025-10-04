from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, mdp_history):
        self.buffer.append(mdp_history)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)