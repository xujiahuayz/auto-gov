import gym
import numpy as np
from test_market import TestMarket


class ProtocolEnv(gym.Env):
    def __init__(self, market: TestMarket):
        self.market = market
        self.action_space = gym.spaces.Discrete(3)  # lower, keep, raise
        self.observation_space = gym.spaces.Box(
            # total_available_funds, total_borrowed_funds, collateral_factor
            low=np.array([0, 0, 0]),
            high=np.array([np.inf, np.inf, 1]),
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.market.reset()
        state = self.market.get_state()
        return state

    def observation(self) -> np.ndarray:
        return self.market.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        state = self.market.get_state()
        collateral_factor = state[2]

        # lower, keep, raise the collateral factor
        if collateral_factor <= 0:
            if action == 0:
                action = 1
            elif action == 2:
                action = 1
        elif collateral_factor >= 1:
            if action == 2:
                action = 1
            elif action == 0:
                action = 1

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


if __name__ == "__main__":
    market = TestMarket()
    env = ProtocolEnv(market)

    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
