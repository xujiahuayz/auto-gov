import random
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from collections import defaultdict
from rl_env import LendingProtocolEnv, Market


@dataclass
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

    # def __repr__(self):
    #     return f"(available funds = {self.total_available_funds}, borrowed funds = {self.total_borrowed_funds})"

    @property
    def utilization_ratio(self) -> float:
        return self.total_borrowed_funds / (
            self.total_available_funds + self.total_borrowed_funds
        )

    def borrow_lend_rates(
        util_rate: float,
        rb_factor: float = 25,
        rs_factor: float = 50,
    ) -> tuple[float, float]:
        """
        calculate borrow and supply rates based on utilization ratio
        with an arbitrarily-set shape
        """

        assert (
            0 <= util_rate < 1
        ), f"utilization ratio must lie in [0,1), but got {util_rate}"

        borrow_rate = util_rate / (rb_factor * (1 - util_rate))
        # initial_borrow_rate / (1 - util_rate) ** EXPONENT
        supply_rate = util_rate / (rs_factor * (1 - util_rate))
        # initial_supply_rate / (1 - util_rate) ** EXPONENT
        return borrow_rate, supply_rate

    @property
    def supply_apy(self) -> float:
        _, rs = self.borrow_lend_rates(self.utilization_ratio)
        return rs

    @property
    def borrow_apy(self) -> float:
        rb, _ = self.borrow_lend_rates(self.utilization_ratio)
        return rb

    @property
    def total_pool_shares(self) -> tuple[float, float]:
        total_i_tokens = sum(self.user_i_tokens.values())
        total_b_tokens = sum(self.user_b_tokens.values())
        return total_i_tokens, total_b_tokens

    @property
    def daily_supplier_multiplier(self) -> float:
        return (1 + self.supply_apy) ** (1 / 365)

    @property
    def daily_borrow_multiplier(self) -> float:
        return (1 + self.borrow_apy) ** (1 / 365)

    def accrue_interest(self):
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


if __name__ == "__main__":
    market = Market()
    print(market)
