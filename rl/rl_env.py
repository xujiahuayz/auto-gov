import gym
import numpy as np


class LendingProtocolEnv(gym.Env):
    def __init__(self, market):
        self.market = market
        self.action_space = gym.spaces.Discrete(3)  # lower, keep, raise
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )

    def reset(self):
        state = self.market.get_state()
        return state

    def step(self, action):
        # lower, keep, raise the collateral factor
        if action == 0:
            self.market.lower_collateral_factor()
        elif action == 1:
            self.market.keep_collateral_factor()
        else:
            self.market.raise_collateral_factor()

        state = self.market.get_state()
        reward = self.market.get_reward()
        done = self.market.is_done()

        return state, reward, done, {}


class Market:
    def __init__(self):
        self.utilization_rate = 0
        self.total_supply = 0
        self.liquidation_threshold = 0
        self.liquidation_discount_factor = 0
        self.collateral_factor = 0
        self.steps = 0
        self.max_steps = 1000

    def get_state(self):
        return np.array(
            [
                self.utilization_rate,
                self.total_supply,
                self.liquidation_threshold,
                self.liquidation_discount_factor,
                self.collateral_factor,
            ]
        )

    def lower_collateral_factor(self):
        self.collateral_factor -= 0.01

    def keep_collateral_factor(self):
        pass

    def raise_collateral_factor(self):
        self.collateral_factor += 0.01

    def get_reward(self):
        # Important!!!!
        # Example reward function
        # we need to use repay?
        reward = self.utilization_rate * self.collateral_factor
        return reward

    def is_done(self):
        self.steps += 1
        if self.steps >= self.max_steps:
            return True
        return False


market = Market()
env = LendingProtocolEnv(market)
