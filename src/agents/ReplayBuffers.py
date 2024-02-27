import numpy as np
from operator import itemgetter


class ReplayBuffer(object):

    def __init__(self, capacity, seed=19834):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.random = np.random.RandomState(seed)

    def push(self, arg):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = arg
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(self.__len__(), size=batch_size, replace=False)
        batch = itemgetter(*idx)(self.memory)
        return batch

    def __len__(self):
        return len(self.memory)
