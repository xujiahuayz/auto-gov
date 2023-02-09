# make sure User is recognized
from __future__ import annotations
from dataclasses import dataclass
from market_env.constants import DEBT_TOKEN_PREFIX, INTEREST_TOKEN_PREFIX
import logging
from typing import Optional

from market_env.utils import PriceDict


class Env:
    def __init__(
        self,
        users: Optional[dict[str, User]] = None,
        prices: Optional[PriceDict] = None,
    ):
        if users is None:
            users = {}

        if prices is None:
            prices = PriceDict({"dai": 1.0, "eth": 2.0})

        self.users = users
        self.prices = prices
        self.step = 0
        self.max_steps = 100_000

    @property
    def prices(self) -> PriceDict:
        return self._prices

    @prices.setter
    def prices(self, value: PriceDict):
        if type(value) is not PriceDict:
            raise TypeError("must use PriceDict type")
        self._prices = value


class User:
    def __init__(self, env: Env, name: str, funds_available: Optional[dict] = None):
        assert name not in env.users, f"User {name} exists"
        self.env = env
        # add the user to the environment
        self.env.users[name] = self

        if funds_available is None:
            funds_available = {"dai": 0.0, "eth": 0.0}
        self.funds_available = funds_available
        self.env = env

        self.name = name

    @property
    def wealth(self) -> float:
        user_wealth = sum(
            value * self.env.prices[asset_name]
            for asset_name, value in self.funds_available.items()
        )
        logging.info(f"{self.name}'s wealth in DAI: {user_wealth}")

        return user_wealth

    def supply_withdraw(self, amount: float, plf: Plf):  # negative for withdrawing
        if self.name not in plf.user_i_tokens:
            plf.user_i_tokens[self.name] = 0

        assert (
            plf.user_i_tokens[self.name] + amount >= 0
        ), "cannot withdraw more i-tokens than you have"

        assert (
            self.funds_available[plf.asset_names] - amount >= 0
        ), "insufficient funds to provide liquidity"

        self.funds_available[plf.asset_names] -= amount

        # update liquidity pool
        plf.total_available_funds += amount

        # update i tokens of the user in the pool registry
        plf.user_i_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.interest_token_name] = plf.user_i_tokens[self.name]

    def borrow_repay(self, amount: float, plf: Plf):
        if self.name not in plf.user_b_tokens:
            plf.user_b_tokens[self.name] = 0

        if plf.borrow_token_name not in self.funds_available:
            self.funds_available[plf.borrow_token_name] = 0

        if plf.user_b_tokens[self.name] + amount < 0:
            raise ValueError("cannot repay more b-tokens than you have")

        if self.funds_available[plf.interest_token_name] * plf.collateral_factor <= (
            amount + self.funds_available[plf.borrow_token_name]
        ):
            raise ValueError(
                "insufficient collateral to get the amount of requested debt tokens"
            )

        # update liquidity pool
        plf.total_borrowed_funds += amount
        plf.total_available_funds -= amount

        # update b tokens of the user in the pool registry
        plf.user_b_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.borrow_token_name] = plf.user_b_tokens[self.name]

        self.funds_available[plf.asset_names] += amount


@dataclass
class Plf:
    env: Env
    initiator: User
    initial_starting_funds: float = 1000
    collateral_factor: float = 0.85
    asset_names: str = "dai"  # you can only deposit and borrow 1 token

    def __post_init__(self):
        self.total_available_funds = self.initial_starting_funds
        self.total_borrowed_funds = 0.0  # start with no funds borrowed

        available_prices = self.env.prices
        self.interest_token_name = INTEREST_TOKEN_PREFIX + self.asset_names
        self.borrow_token_name = DEBT_TOKEN_PREFIX + self.asset_names

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

    def __repr__(self):
        return f"(available funds = {self.total_available_funds}, borrowed funds = {self.total_borrowed_funds})"

    @property
    def utilization_ratio(self) -> float:
        return self.total_borrowed_funds / (
            self.total_available_funds + self.total_borrowed_funds
        )

    @property
    def supply_apy(self) -> float:
        _, rs = self.borrow_lend_rates(util_rate=self.utilization_ratio)
        return rs

    @property
    def borrow_apy(self) -> float:
        rb, _ = self.borrow_lend_rates(util_rate=self.utilization_ratio)
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

    def borrow_lend_rates(
        self,
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
        supply_rate = util_rate / (rs_factor * (1 - util_rate))

        return borrow_rate, supply_rate

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
