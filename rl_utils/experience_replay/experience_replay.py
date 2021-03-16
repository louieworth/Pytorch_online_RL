import numpy as np
import random

"""
define the replay buffer and corresponding algorithms 
"""

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory_size = capacity
        self.buffer = []
        self.idx = 0

    # add the sample
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if self.idx >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.idx] = data
        self.idx = (self.idx + 1) % self.memory_size

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def size(self):
        return len(self.buffer)
"""
    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
"""