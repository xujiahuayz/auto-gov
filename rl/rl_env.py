import sys
from typing import Union
import gym
import numpy as np
from rl.test_market import TestMarket
from market_env.env import DefiEnv, PlfPool, User, PriceDict


class ProtocolEnv(gym.Env):
    def __init__(self, defi_env: DefiEnv):
        self.defi_env = defi_env
        self.action_space = gym.spaces.Discrete(3)  # lower, keep, raise
        self.observation_space = gym.spaces.Box(
            # self.total_available_funds,
            # self.total_borrowed_funds,
            # self.collateral_factor,
            # self.total_i_tokens,
            # self.total_b_tokens,
            low=np.array([0, -np.inf, -np.inf, 0, 0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        # self.market.reset()
        self.defi_env.reset()
        state = self.defi_env.get_state()
        return state

    def observation(self) -> np.ndarray:
        return self.defi_env.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        state = self.defi_env.get_state()
        collateral_factor = state[2]

        # # constrain the action
        # if collateral_factor <= 0:
        #     if action == 0:
        #         action = 1
        #     elif action == 2:
        #         action = 1
        # elif collateral_factor >= 1:
        #     if action == 2:
        #         action = 1
        #     elif action == 0:
        #         action = 1

        # lower, keep, raise the collateral factor
        if action == 0:
            self.defi_env.lower_collateral_factor()
        elif action == 1:
            self.defi_env.keep_collateral_factor()
        elif action == 2:
            self.defi_env.raise_collateral_factor()

        state = self.defi_env.get_state()
        reward = self.defi_env.get_reward()
        done = self.defi_env.is_done()

        return state, reward, done, {}


class DefiProtocolEnv(gym.Env):
    def __init__(self, defi_env: DefiEnv):
        self.defi_env = defi_env
        self.action_space = gym.spaces.Discrete(3)  # lower, keep, raise
        self.observation_space = gym.spaces.Box(
            # total_available_funds, total_borrowed_funds, collateral_factor
            low=np.array([0, 0, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        # self.market.reset()
        state = self.defi_env.get_state()
        return state

    def observation(self) -> np.ndarray:
        return self.defi_env.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        state = self.defi_env.get_state()
        collateral_factor = state[2]

        # # constrain the action
        # if collateral_factor <= 0:
        #     if action == 0:
        #         action = 1
        #     elif action == 2:
        #         action = 1
        # elif collateral_factor >= 1:
        #     if action == 2:
        #         action = 1
        #     elif action == 0:
        #         action = 1

        # lower, keep, raise the collateral factor
        if action == 0:
            self.defi_env.lower_collateral_factor()
        elif action == 1:
            self.defi_env.keep_collateral_factor()
        elif action == 2:
            self.defi_env.raise_collateral_factor()

        state = self.defi_env.get_state()
        reward = self.defi_env.get_reward()
        done = self.defi_env.is_done()

        return state, reward, done, {}


if __name__ == "__main__":
    market = TestMarket()
    env = ProtocolEnv(market)

    print(env.reset())
