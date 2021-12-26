import numpy as np
import torch
from model import Net
from datetime import datetime
from utils import select_actions, evaluate_actions, discount_with_dones
import os

class Agent(object):
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args

        self.net = Net(self.env.action_space.n)
        if self.args.cuda:
            self.net.cuda()

        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.args.lr, eps=self.args.eps)
        if not os.path.exists(self.args.save_path):
            os.mkdir(self.args.save_path)
        self.model_path = self.args.save_path + self.args.env_name + '/'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)


