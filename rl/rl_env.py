import gym
import numpy as np
from market_env import Market


class Market:
    def __init__(self):
        self.liquidation_threshold: float = 0
        self.liquidation_discount_factor: float = 0
        self.collateral_factor: float = 0.85
        self.user_i_token: float = 0  # total supply
        self.user_b_token: float = 0  # total borrow
        self.utilization_rate: float = self.user_b_token / self.user_i_token
        self.steps = 0
        self.max_steps = 10000
        self.cumulative_protocol_earning: float = 0
        self.this_step_protocol_earning: float = 0

    def get_state(self) -> np.ndarray:
        return np.array(
            [
                self.utilization_rate,
                self.total_supply,
                self.liquidation_threshold,
                self.liquidation_discount_factor,
                self.collateral_factor,
            ]
        )

    def update_market(self) -> None:
        pass

    def lower_collateral_factor(self) -> None:
        self.collateral_factor -= 0.01
        self.update_market()

    def keep_collateral_factor(self) -> None:
        self.update_market()

    def raise_collateral_factor(self) -> None:
        self.collateral_factor += 0.01
        self.update_market()

    def get_reward(self) -> float:
        # Important!!!!
        # Example reward function
        reward = self.this_step_protocol_earning
        return reward

    def is_done(self) -> bool:
        self.steps += 1
        if self.steps >= self.max_steps:
            return True
        return False

    def accrue_interest(self):
        # update this!!!
        for user_name in self.user_i_tokens:
            user_funds = self.env.users[user_name].funds_available

            # distribute i-token
            user_funds[self.interest_token_name] *= self.daily_supplier_multiplier

            # update i token register
            self.user_i_tokens[user_name] = user_funds[self.interest_token_name]

        for user_name in self.user_b_tokens:
            user_funds = self.env.users[user_name].funds_available

            # distribute b-token
            user_funds[self.borrow_token_name] *= self.daily_borrow_multiplier

            # update b token register
            self.user_b_tokens[user_name] = user_funds[self.borrow_token_name]


class LendingProtocolEnv(gym.Env):
    def __init__(self, market: Market):
        self.market = market
        self.action_space = gym.spaces.Discrete(3)  # lower, keep, raise
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        state = self.market.get_state()
        return state

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
