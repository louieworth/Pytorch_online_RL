from types import LambdaType
from typing import List
import numpy as np 
import pandas as pd 
import time 

from pandas.core.construction import is_empty_data

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9  # greedy-policy
ALPHA = 0.1 # learning rate
LAMBDA = 0.9 # discounting factor
MAX_EPSILON = 12 # max epsilons
FRENSH_TIME = 0.03

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or (state_actions == 0).all():
        action =  np.random.choice(ACTIONS)
    else:
        action = state_actions.idxmax()
    return action

def get_env_feedback(S, A):
    # this is how the agent interact with the environment
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S 
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (
            episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = '*'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRENSH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPSILON):
        step_counter = 0
        S = 0 #initial state
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table) # epsilon greedy 
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1
            # print('Q_table\n', q_table)
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ_table: \n: ', q_table)
    

