import numpy as np
import gym
from AC import Agent
from utils import plotLearning
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

if __name__ == "__main__":
    agent = Agent(alpha=5e-5, beta=1e-5, input_dim=[2], gamma=0.99,
                  layer1_size=256, layer2_size=256)

    env = gym.make('MountainCarContinuous-v0')
    scores = []
    num_episodes = 10
    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = np.array(agent.choose_action(observation)).reshape((1, ))
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        scores.append(score)
        print('Episode: {}, | Score: {:.2f}'.format(i, score))
        writer.add_scalar('Accuarcy/loss', score, i)

    # filename = 'Mountaincar-continous.png'
    # plotLearning(scores, filename, window=20)









