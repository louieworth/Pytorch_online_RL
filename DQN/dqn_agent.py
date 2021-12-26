import sys
import numpy as np
from model import net
from utils import linear_schedule, select_actions, reward_recorder
from rl_utils.experience_replay.experience_replay import ReplayBuffer

import torch
import torch.nn as nn
from datetime import datetime
import os
import copy

# define the dqn agent
class dqn_agent:
    def __init__(self, env, args):
        # define hyper parameters
        self.env = env
        self.args = args
        self.criterion = nn.MSELoss()
        # define network
        self.net = net(self.env.action_space.n, self.args.use_dueling)
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
            os.mkdir(self.model_path)

    def learn(self):
        episode_reward = reward_recorder()
        state = np.array(self.env.reset())
        td_loss = 0
        for timestep in range(self.args.total_timesteps):
            explore_eps = self.exploration_schedule.get_value(timestep)
            with torch.no_grad():
                state_tensor = self._get_tensors(state)
                action_value = self.net(state_tensor)
            # select action
            action = select_actions(action_value, explore_eps)
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.array(next_state)
            # append samples
            self.buffer.add(state, action, reward, next_state, float(done))
            state = next_state
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
            if done and episode_reward.num_episodes % self.args.display_interval == 0:
                print("[{}] Frames: {}, Episode: {}, Mean{:.3f}, Loss {:.3f}".format(datetime.now(), timestep, episode_reward.num_episodes,\
                                                                                     episode_reward.mean, td_loss))
                torch.save(self.net.state_dict(), self.model_path + '/model.pt')
    # update network
    # update the network
    def _update_network(self, samples):
        states, actions, rewards, states_next, dones = samples
        # convert the data to tensor
        states = self._get_tensors(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        states_next = self._get_tensors(states_next)
        dones = torch.tensor(1 - dones, dtype=torch.float32).unsqueeze(-1)
        # convert into gpu
        if self.args.cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()
        # calculate the target value
        with torch.no_grad():
            # if use the double network architecture
            if self.args.use_double_net:
                """
                Current Q-network w is used to choose action 
                older Q-network w' is used to evaluate action
                """
                # choose action
                q_value_ = self.net(states_next)
                action_max_idx = torch.argmax(q_value_, dim=1, keepdim=True)
                # evaluate action
                target_action_value = self.target_net(states_next)
                target_action_max_value = target_action_value.gather(1, action_max_idx)
            else:
                target_action_value = self.target_net(states_next)  # torch.size([32, 6])
                target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)  # torch.size([32,1])
        target_value = rewards + self.args.gamma * target_action_max_value * dones
        # get the real q value
        action_value = self.net(states)
        predict_value = action_value.gather(1, actions) # based on primary actions
        loss = self.criterion(predict_value, target_value)
        # start to update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_tensors(self, states):
        states = np.array(states)
        if states.ndim == 3:
            states = np.transpose(states, (2, 0, 1))
            states = np.expand_dims(states, 0) # TODO
        elif states.ndim == 4:
            states = np.transpose(states, (0, 3, 1, 2))
        states = torch.FloatTensor(states)
        if self.args.cuda:
            states = states.cuda()
        return states




