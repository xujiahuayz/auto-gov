import gym
from dqn_gov import Agent
from utils2 import plot_learning_curve
import numpy as np
from test_market import TestMarket
from rl_env import ProtocolEnv

if __name__ == "__main__":
    # initialize market and environment
    market = TestMarket()
    env = ProtocolEnv(market)
    print(env.reset())
    print(env.step(2))
    print(env.reset())
