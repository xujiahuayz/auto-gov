import gym
import numpy as np

from market_env.env import DefiEnv
from rl.test_market import TestMarket


class ProtocolEnv(gym.Env):
    def __init__(self, defi_env: DefiEnv):
        self.defi_env = defi_env
        num_pools = len(defi_env.plf_pools)
        self.observation_space = gym.spaces.Box(
            # self.total_available_funds,
            # self.total_borrowed_funds,
            # self.total_i_tokens,
            # self.total_b_tokens,
            # self.collateral_factor,
            # self.utilization_ratio,
            # self.supply_apy,
            # self.borrow_apy,
            low=np.array([0, -np.inf, 0, 0, -np.inf, 0, 0, 0] * num_pools),
            high=np.array(
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
                * num_pools
            ),
            dtype=np.float32,
        )
        num_action = defi_env.num_action_pool**num_pools
        self.action_space = gym.spaces.Discrete(num_action)  # lower, keep, raise

    def reset(self) -> np.ndarray:
        # self.market.reset()
        self.defi_env.reset()
        state = self.defi_env.get_state()
        return state

    def observation(self) -> np.ndarray:
        return self.defi_env.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        state = self.defi_env.get_state()

        self.defi_env.update_collateral_factor(action)

        state = self.defi_env.get_state()
        reward = self.defi_env.get_reward()
        done = self.defi_env.is_done()

        return state, reward, done, {}


class TestProtocolEnv(gym.Env):
    def __init__(self, defi_env: TestMarket):
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
    pass
