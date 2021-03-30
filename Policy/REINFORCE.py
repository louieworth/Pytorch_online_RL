import argparse
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Pytorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar="G",
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=34, metavar='G',
                    help = "random seed (default: 34)")
parser.add_argument('--render', action='store_true',
                    help='render the environment (default: True)')
parser.add_argument('--log-interval', type=int, default=10, metavar="N",
                    help='interval between training status logs (default: 10)')
parser.add_argument('--env_name', default='CartPole-v0',
                    help='the environment name from gym (default: Cartpole-v0)')
parser.add_argument('--lr', type=float, default=0.01, metavar="LR",
                    help='learning rate')

args = parser.parse_args()

env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.layer1(x))
        action_score = F.relu(self.layer2(x))
        return action_score

policy = Policy(4, 2)
optimizer = optim.Adam(policy.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()
def select_action(state):
    state = torch.unsqueeze(torch.FloatTensor(state), 0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma  * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(- log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.saved_log_probs[: ]
    del policy.rewards[:]

def main():
    episodes = []
    rewards = []
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        # TODO
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            episodes.append(i_episode)
            rewards.append(running_reward)
            print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            plt.plot(episodes, rewards)
            plt.show()
            break


if __name__ == '__main__':
    main()


