import gym
import numpy as np

from market_env.env import DefiEnv


class ProtocolEnv(gym.Env):
    def __init__(
        self,
        defi_env: DefiEnv,
    ):
        self.defi_env = defi_env
        num_pools = len(defi_env.plf_pools)
        self.observation_space = gym.spaces.Box(
            # self.total_available_funds,
            # self.reserve,
            # self.total_i_tokens,
            # self.total_b_tokens,
            # self.collateral_factor,
            # self.utilization_ratio,
            # self.supply_apy,
            # self.borrow_apy,
            # self.env.prices[self.asset_name],
            # self.asset_volatility[self.env.step],
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * num_pools),
            high=np.array(
                [
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    1,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ]
                * num_pools
            ),
            dtype=np.float32,
        )
        num_action = defi_env.num_action_pool**num_pools
        self.action_space = gym.spaces.Discrete(num_action)  # lower, keep, raise

    def reset(self) -> np.ndarray:
        # self.market.reset()
        self.defi_env.reset()
        return self.defi_env.get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.defi_env.act_update_react(action)
        state = self.defi_env.get_state()
        reward = self.defi_env.get_reward()
        done = self.defi_env.is_done()

        return state, reward, done, {}
