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

    def __post_init__(self):
        self.total_available_funds = self.initial_starting_funds
        self.total_borrowed_funds = 0.0  # start with no funds borrowed

        available_prices = self.env.prices
        # self.interest_token_name = INTEREST_TOKEN_PREFIX + self.asset_names
        # self.borrow_token_name = DEBT_TOKEN_PREFIX + self.asset_names

        assert (
            self.asset_names in self.initiator.funds_available
            and self.initiator.funds_available[self.asset_names]
            >= self.initial_starting_funds
        ), "insufficient funds"

        # deduct funds from user balance
        self.initiator.funds_available[self.asset_names] -= self.initial_starting_funds

        self.user_i_tokens = {self.initiator.name: self.initial_starting_funds}

        self.user_b_tokens = {self.initiator.name: 0.0}

        # add interest-bearing token into initiator's wallet
        self.initiator.funds_available[
            self.interest_token_name
        ] = self.initial_starting_funds
        self.initiator.funds_available[self.borrow_token_name] = 0

        # if reward token is a new token, then initiate price with 0
        reward_token_name = self.reward_token_name
        if reward_token_name not in available_prices:
            available_prices[self.reward_token_name] = 0

    def __repr__(self):
        return f"(available funds = {self.total_available_funds}, borrowed funds = {self.total_borrowed_funds})"

    @property
    def utilization_ratio(self) -> float:
        return self.total_borrowed_funds / (
            self.total_available_funds + self.total_borrowed_funds
        )

    @property
    def supply_apy(self) -> float:
        _, rs = borrow_lend_rates(self.utilization_ratio)
        return rs

    @property
    def borrow_apy(self) -> float:
        rb, _ = borrow_lend_rates(self.utilization_ratio)
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

    def get_user_pool_fraction(self, user_name: str) -> tuple[float, float]:
        if user_name not in self.user_i_tokens:
            self.user_i_tokens[user_name] = self.env.users[user_name].funds_available[
                self.interest_token_name
            ] = 0.0
        i_token_fraction = self.user_i_tokens[user_name] / self.total_pool_shares[0]

        if user_name not in self.user_b_tokens:
            self.user_b_tokens[user_name] = self.env.users[user_name].funds_available[
                self.borrow_token_name
            ] = 0.0

        assert 0 <= self.user_b_tokens[user_name] <= self.total_pool_shares[1]

        if self.total_pool_shares[1] == 0:
            b_token_fraction = 0
        else:
            b_token_fraction = self.user_b_tokens[user_name] / self.total_pool_shares[1]

        return i_token_fraction, b_token_fraction

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
