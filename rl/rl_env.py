import gym
import numpy as np
from market_env import Market


class LendingProtocolEnv(gym.Env):
    def __init__(self, market: Market):
        self.market = market
        self.action_space = gym.spaces.Discrete(3)  # lower, keep, raise
        self.observation_space = gym.spaces.Box(
            # utilization_ratio, total_supply, collateral_factor
            low=np.array([0, 0, 0]),
            high=np.array([1, np.inf, 1]),
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        state = self.market.get_state()
        return state

    def observation(self) -> np.ndarray:
        return self.market.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
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


if __name__ == "__main__":
    market = Market()
    env = LendingProtocolEnv(market)

    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
