from collections import deque
import random


class ReplayMemory:
    """
    Experience Replay Buffer using a FIFO deque.
    Stores (state, action, reward, next_state, terminated) tuples.
    When full, the oldest experience is automatically removed from the front.
    """

    def __init__(self, max_length, seed=None):
        self.memory = deque(maxlen=max_length)
        if seed is not None:
            random.seed(seed)

    def append(self, experience):
        """Add a new experience tuple to the memory."""
        self.memory.append(experience)

    def sample(self, sample_size):
        """Return a random batch of experiences."""
        return random.sample(self.memory, sample_size)

    def __len__(self):
        """Return current number of stored experiences."""
        return len(self.memory)
