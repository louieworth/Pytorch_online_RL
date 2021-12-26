import torch
import numpy as np
from torch.distributions.categorical import Categorical

def select_actions(pi, deterministic=False):
    model = Categorical(pi)
    if deterministic:
        return torch.argmax(pi, dim=1).item()
    else:
        return model.sample().unsqueeze(-1)

def evaluate_actions(pi, actions):
    model = Categorical(pi)
    return model.log_prob(actions.squeeze(-1)).unsqueeze(-1), model.entropy().mean()

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1 - done)
        discounted.append(r)

    return discounted[::-1]