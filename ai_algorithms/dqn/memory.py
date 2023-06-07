from collections import deque
from .transition import Transition
import random


class ReplayMemory:
    def __init__(self, capacity=1000):
        self.capacity=capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)