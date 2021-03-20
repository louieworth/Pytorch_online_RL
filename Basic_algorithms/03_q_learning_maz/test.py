from maze_env import Maze
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # choose action based on environment
            action = RL.choose_actions(str(observation))

            # Rl take action and get the next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from the transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
    print('Game Over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
env.mainloop()




