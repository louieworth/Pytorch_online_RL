import gym
import os
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(2)
torch.manual_seed(2)

state_space = env.observation_space.shape[0]
action_space = env.observation_space.n

lr = 0.01
gamma = 0.99
episodes = 20000
render = False
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SaveAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, 32)
        self.value = nn.Linear(32, 1)
        self.action = nn.Linear(32, action_space)

        self.save_actions = []
        self.rewards = []
        os.makedirs('./AC_cartPole-v0', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        state_value = self.value(x)
        action = self.action(x)

        return F.softmax(action, dim=-1), state_value

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=lr)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.save_actions.append(SavedAction(m.log_prob(action), state_value))

def update_network():
    R = 0
    save_actions = model.save_actions
    policy_loss = []
    value_loss = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob, value), r in zip(save_actions, rewards):
        reward = r - value.item()
        policy_loss.append(- log_prob * reward)
        value_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.save_actions[:]

