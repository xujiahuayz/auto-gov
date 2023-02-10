# make sure User is recognized
from __future__ import annotations
from dataclasses import dataclass
from market_env.constants import DEBT_TOKEN_PREFIX, INTEREST_TOKEN_PREFIX
import logging
from typing import Optional
import numpy as np
import logging

from market_env.utils import PriceDict

# set logging level
logging.basicConfig(level=logging.INFO)

SAFETY_MARGIN = 0.05


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

    @property
    def is_done(self) -> bool:
        """
        Returns True if step reaches max_steps, otherwise False
        """
        self.step += 1
        if self.step >= self.max_steps:
            return True
        return False


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
        logging.info(f"{self.name}'s wealth in USD: {user_wealth}")

        return user_wealth

    def __repr__(self) -> str:
        return f"name: {self.name}, funds available: {self.funds_available}, wealth: {self.wealth}"

    def _supply_withdraw(self, amount: float, plf: Plf) -> None:
        """
        Supply (amount > 0) or withdraw (amount < 0) funds to the liquidity pool
        """
        # set default values for user_i_tokens and asset balance if they don't exist
        plf.user_i_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf.asset_name, 0)

        if amount > 0:
            amount = min(amount, self.funds_available[plf.asset_name])
            # add logging
            logging.info(f"supplying {amount} {plf.asset_name}")
        else:
            amount = max(amount, -plf.user_i_tokens[self.name])
            # add logging
            logging.info(f"withdrawing {amount} {plf.asset_name}")

        self.funds_available[plf.asset_name] -= amount

        # update liquidity pool
        plf.total_available_funds += amount

        # update i tokens of the user in the pool registry
        plf.user_i_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.interest_token_name] = plf.user_i_tokens[self.name]

    def _borrow_repay(self, amount: float, plf: Plf) -> None:
        # set default values for user_b_tokens and funds_available if they don't exist

        plf.user_b_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf.borrow_token_name, 0)
        self.funds_available.setdefault(plf.asset_name, 0)
        self.funds_available.setdefault(plf.interest_token_name, 0)

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

        self.funds_available[plf.asset_name] += amount

    def reactive_action(self, plf: Plf) -> None:
        """
        supply funds to the liquidity pool in response to market conditions
        """

        self.funds_available.setdefault(plf.asset_name, 0)
        self.funds_available.setdefault(plf.interest_token_name, 0)
        self.funds_available.setdefault(plf.borrow_token_name, 0)

        # NOTE: this is only looking at one market, not considering user's other assets
        # TODO: add a check for the user's other assets
        # check loan health
        existing_borrow = self.funds_available[plf.borrow_token_name]
        existing_supply = self.funds_available[plf.interest_token_name]

        max_borrow = existing_supply * plf.collateral_factor / (1 + SAFETY_MARGIN)

        if max_borrow > existing_borrow:  # healthy loan
            # can deposit all existing funds in the pool
            self._supply_withdraw(self.funds_available[plf.asset_name], plf)
            # can borrow up to the buffer
            new_loan = (
                self.funds_available[plf.interest_token_name] * plf.collateral_factor
            ) / (1 + SAFETY_MARGIN)
            self._borrow_repay(new_loan - existing_borrow, plf)
        else:  # unhealthy loan
            self._borrow_repay(max_borrow - existing_borrow, plf)


class Plf:
    def __init__(
        self,
        env: Env,
        initiator: User,
        initial_starting_funds: float = 1000,
        collateral_factor: float = 0.85,
        asset_name: str = "dai",
    ) -> None:
        """
        :param env: the environment in which the liquidity pool is operating
        :param initiator: the user who is initiating the liquidity pool
        :param initial_starting_funds: the initial amount of funds the initiator is putting into the liquidity pool
        :param collateral_factor: the collateral factor of the liquidity pool
        :param asset_name: the name of the asset that the liquidity pool is operating with
        """

        if collateral_factor > 1 or collateral_factor < 0:
            raise ValueError("collateral factor must be between 0 and 1")
        initiator.funds_available.setdefault(asset_name, 0)
        if initiator.funds_available[asset_name] < initial_starting_funds:
            raise ValueError("insufficient funds to start liquidity pool")

        self.env = env
        self.initiator = initiator
        self.initial_starting_funds = initial_starting_funds
        self._collateral_factor = collateral_factor
        self.asset_name = asset_name

        self.previous_profit: float = 0.0
        self.previous_reward: float = 0.0

        # start with no funds borrowed, actual underlying that's been borrowed, not the interest-accruing debt tokens
        self.total_borrowed_funds = 0.0

        self.interest_token_name = INTEREST_TOKEN_PREFIX + self.asset_name
        self.borrow_token_name = DEBT_TOKEN_PREFIX + self.asset_name

        # deduct funds from user balance
        self.initiator.funds_available[self.asset_name] -= self.initial_starting_funds

        # actual underlying that's still available, not the interest-bearing tokens
        self.total_available_funds = self.initial_starting_funds

        self.user_i_tokens = {self.initiator.name: self.initial_starting_funds}

        self.user_b_tokens = {self.initiator.name: 0.0}

        # add interest-bearing token into initiator's wallet
        self.initiator.funds_available[
            self.interest_token_name
        ] = self.initial_starting_funds
        self.initiator.funds_available[self.borrow_token_name] = 0

    def __repr__(self) -> str:
        return f"{self.asset_name} PLF: (available funds = {self.total_available_funds}, borrowed funds = {self.total_borrowed_funds}, profit = {self.profit})"

    @property
    def collateral_factor(self):
        return self._collateral_factor

    @collateral_factor.setter
    def collateral_factor(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("collateral factor must be between 0 and 1")
        self._collateral_factor = value

    # actions
    def lower_collateral_factor(self) -> None:
        self.collateral_factor -= 0.01
        self.update_market()

    def keep_collateral_factor(self) -> None:
        self.update_market()

    def raise_collateral_factor(self) -> None:
        self.collateral_factor += 0.01
        self.update_market()

    def update_market(self) -> None:
        # self.previous_profit = self.profit

        self.accrue_daily_interest()
        for user in self.env.users.values():
            user.reactive_action(self)

    def get_state(self) -> np.ndarray:
        return np.array(
            [
                self.total_available_funds,
                self.total_borrowed_funds,
                self.collateral_factor,
            ]
        )

    def get_profit(self) -> float:
        previous_profit = self.previous_profit
        self.previous_profit = self.profit
        return self.profit - previous_profit

    def get_reward(self) -> float:
        """
        get the difference between the profit gained from this episode and the profit gained from the previous episode
        """
        this_reward = self.get_profit()
        reward_diff = this_reward - self.previous_reward
        self.previous_profit = this_reward
        return reward_diff

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
    def total_i_tokens(self) -> float:
        return sum(self.user_i_tokens.values())

    @property
    def total_b_tokens(self) -> float:
        return sum(self.user_b_tokens.values())

    @property
    def profit(self) -> float:
        return self.total_b_tokens + self.total_available_funds - self.total_i_tokens

    @property
    def daily_supplier_multiplier(self) -> float:
        return (1 + self.supply_apy) ** (1 / 365)

    @property
    def daily_borrow_multiplier(self) -> float:
        return (1 + self.borrow_apy) ** (1 / 365)

    def borrow_lend_rates(
        self,
        util_rate: float,
        rb_factor: float = 20,
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

    def accrue_daily_interest(self):
        """
        accrue interest to all users in the pool
        record profit
        """
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
    # initialize environment
    env = Env(prices=PriceDict({"tkn": 1}))
    Alice = User(name="alice", env=env, funds_available={"tkn": 2_000})
    plf = Plf(
        env=env,
        initiator=Alice,
        initial_starting_funds=1000,
        asset_name="tkn",
        collateral_factor=0.8,
    )
    print(f"**Alice {Alice} \n")
    print(f"++{plf} \n")

    # Alice react first
    Alice.reactive_action(plf)
    print(f"**Alice {Alice} \n")
    print(f"++{plf} \n")

    # then the market update
    plf.raise_collateral_factor()
    plf.update_market()
    Alice.reactive_action(plf)
    print(f"**Alice {Alice} \n")
    print(f"++{plf} \n")
