import numpy as np
from numpy.core.arrayprint import printoptions
import pandas as pd
import matplotlib.pyplot as plt
import time 

from cyberbrain import trace

ALPHA = 0.1
GAMMA = 0.95
EPSILION = 0.9
N_STATE = 6
ACTIONS = ['left', 'right']
MAX_EPISODES = 50
FRESH_TIME = 0.1

def build_q_table(n_states, actions):
    q_table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        np.arange(n_states),
        actions
    )
    print(q_table)
    return q_table
@trace 
def choose_action(state, q_table):
    state_actions = q_table.loc[state, :]
    if (np.random.uniform() > EPSILION) or (state_actions==0).all():
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name
@trace 
def get_env_feedback(state, action):
    if action == "right":
        if state == N_STATE - 2:
            next_state = "terminal"
            reward = 1
        else:
            next_state = state + 1
            reward = -0.5
    
    else:
        if state == 0:
            next_state = 0 
        else:
            next_state = state - 1
        reward = -0.5
    return next_state, reward

def update_env(state,episode, step_counter):
    env = ['-'] *(N_STATE-1)+['T']
    if state =='terminal':
        print("Episode {}, the total step is {}".format(episode+1, step_counter))
        final_env = ['-'] *(N_STATE-1)+['T']
        return True, step_counter
    else:
        env[state]='*'
        env = ''.join(env)
        print(env)
        time.sleep(FRESH_TIME)
        return False, step_counter
@trace  
def sarsa_learning():
    q_table = build_q_table(N_STATE, ACTIONS)
    step_counter_times = []
    for episode in range(MAX_EPISODES):
        state = 0
        is_terminal = False
        step_counter = 0
        update_env(state, episode, step_counter)
        while not is_terminal:
            action = choose_action(state, q_table)
            q_predict = q_table.loc[state, action]
            next_state, reward = get_env_feedback(state, action)
            if next_state != 'terminal':
                next_action = choose_action(next_state, q_table)
                q_target = reward + GAMMA * q_table.loc[next_state, next_action]
                q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            else:
                next_action = action
                is_terminal = True

            state = next_state
            is_terminal, steps = update_env(state, episode, step_counter+1)
            step_counter += 1
            if is_terminal:
                step_counter_times.append(steps)
    return q_table, step_counter_times
          
if __name__ == '__main__':
    q_table, step_counter_times = sarsa_learning()
    print(f"Q table \n {q_table}\n")
    print("end")

    plt.plot(step_counter_times, 'g-')
    plt.ylabel("steps")
    plt.show()
    print(f"The step counter_times is {step_counter_times}")