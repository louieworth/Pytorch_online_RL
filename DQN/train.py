import sys
from arguments import get_args
from rl_utils.env_wrapper.create_env import create_single_env
from rl_utils.logger import logger, bench
from rl_utils.seeds.seeds import set_seeds
from dqn_agent import dqn_agent
import os
import numpy as np

if __name__ == "__main__":
    args = get_args()
    env = create_single_env(args)
    set_seeds(args)
    dqn_trainer = dqn_agent(env, args)
    dqn_trainer.learn()
    env.close()