import sys
import numpy as np
from model import net
from utils import linear_schedule, select_actions, reward_recorder
from rl_utils.experience_replay.experience_reaply import ReplayBuffer

import torch
import torch.nn as nn
from datetime import datetime
import os
import copy

# define the dqn agent
class dqn_agent:
    def __init__(self, env, args): # TODO, how to import args from arguments
        # define hyper parameters
        self.env = env
        self.args = args
        self.loss_func = nn.MSELoss()
        # define network
        self.net = net(self.env.action_action_space.n, self.args.use_dueling_action)
        self.target_net = copy.deepcopy(self.net)
        self.target_net.load_state_dict(self.net.state_dict())
        if self.args.cuda:
            self.net.cuda()
            self.target_net.cuda()
        # define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.buffer = ReplayBuffer(self.args.batch_size)
        # define the linear schedule of the exploration
        # TODO
        self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), \
                                                    self.args.final_ratio, self.args.init_ratio)
        # create the folder to save the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # set environment folder
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.args.env_name)

    def learn(self):
        episode_reward = reward_recorder()
        action = np.array(self.env.reset())
        td_loss = 0
        for timestep in range(self.args.total_timesteps):
            explore_eps = self.exploration_schedule.get_value(timestep)
            with torch.no_grad():
                action_tensor = self._get_tensors(action)
                action_value = self.net(action_tensor)
            # select action
            action = select_actions(action_value, explore_eps)
            next_action, reward, done, _ = self.env.step(action)
            action = next_action
            # add the reward
            episode_reward.add_rewards(reward)
            if done:
                action = np.array(self.env.reset())
                episode_reward.start_new_episode()
            # sample the samples from the replay buffer
            if timestep > self.args.learning_starts and timestep % self.args.train_freq == 0:
                batch_samples = self.buffer.sample(self.args.batch_size)
                td_loss = self._update_network(batch_samples)
            if timestep > self.args.learning_starts and timestep % self.args.target_network_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())
            if done and episode_reward.num_episodes % self.args.display_freq == 0:
                print("[{}] Frames: {}, Episode: {}, Mean{:.3f}, Loss {:.3f}".format(datetime.now(), timestep, episode_reward.num_episodes\
                                                                                     episode_reward.mean, td_loss))
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')

    # update network
    def _update_netwrok(self, samples):
        states, actions, rewards, next_states, dones = samples
        # convert to tensor
        states = self._get_tensors(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1) #TODO
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = self._get_tensors(next_states)
        dones = torch.FloatTensor(1-dones).unsqueeze(-1)
        # convert to gpu
        if self.args.cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()
        # calculate the target value
        with torch.no_grad():
            # if use the double network architecture
            if self.args.use_double_net:
                q_value = self.net(states)
                action_max_id = torch.argmax(q_value, dim=1, keepdim=True)
                target_action_value = self.target_net(next_states)
                target_action_max_value = target_action_value.gather(1, action_max_id)
            else:
                target_action_value = self.target_net(next_states)
                target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)
        # target
        target_value = rewards + self.args.gamma * target_action_max_value * dones
        action_value = self.net(actions)
        predict_value = action_value.gather(1, actions)
        loss = self.loss_func(target_value, predict_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_tensors(self, states):
        if states.ndim == 3:
            states = np.transpose(states, (2, 0, 1))
            states = np.expand_dims(states, 0) # TODO
        elif states.ndim == 4:
            states = np.transpose(states, (0, 3, 1, 2))
        states = torch.FloatTensor(states)
        if self.args.cuda:
            states = states.cuda()
        return states




