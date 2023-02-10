# pylint: disable=missing-module-docstring,missing-function-docstring
# make sure User is recognized
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from market_env.constants import DEBT_TOKEN_PREFIX, INTEREST_TOKEN_PREFIX
from market_env.utils import PriceDict

# set logging level
logging.basicConfig(level=logging.INFO)


class DefiEnv:
    def __init__(
        self,
        users: Optional[dict[str, User]] = None,
        prices: Optional[PriceDict] = None,
        plf_pools: Optional[dict[str, PlfPool]] = None,
    ):
        if users is None:
            users = {}

        if plf_pools is None:
            plf_pools = {}

        if prices is None:
            prices = PriceDict({"dai": 1.0, "eth": 2.0})

        self.users = users
        self.prices = prices
        self.plf_pools = plf_pools
        self.step = 0
        self.max_steps = 30

    @property
    def prices(self) -> PriceDict:
        return self._prices

    @prices.setter
    def prices(self, value: PriceDict):
        if not isinstance(value, PriceDict):
            raise TypeError("must use PriceDict type")
        self._prices = value

    def lower_collateral_factor(self) -> None:
        self._apply_to_all_pools(PlfPool.lower_collateral_factor)

    def keep_collateral_factor(self) -> None:
        self._apply_to_all_pools(PlfPool.keep_collateral_factor)

    def raise_collateral_factor(self) -> None:
        self._apply_to_all_pools(PlfPool.raise_collateral_factor)

    def get_reward(self) -> float:
        return sum(self._apply_to_all_pools(PlfPool.get_reward))

    def get_state(self) -> np.ndarray:
        return np.concatenate(self._apply_to_all_pools(PlfPool.get_state))

    def is_done(self) -> bool:
        """
        Returns True if step reaches max_steps, otherwise False
        """
        self.step += 1
        return self.step >= self.max_steps

    def reset(self):
        for user in self.users.values():
            user.reset()
        self._apply_to_all_pools(PlfPool.reset)

    def _apply_to_all_pools(self, func):
        return [func(pool) for pool in self.plf_pools.values()]


class User:
    def __init__(
        self,
        env: DefiEnv,
        name: str,
        safety_margin=0.05,
        funds_available: Optional[dict] = None,
    ):
        assert name not in env.users, f"User {name} exists"
        self.env = env
        # add the user to the environment
        self.env.users[name] = self

        if funds_available is None:
            funds_available = {"dai": 0.0, "eth": 0.0}
        self._initial_funds_available = funds_available.copy()
        self.funds_available = funds_available
        self.env = env

        self.name = name
        self.safety_margin = safety_margin
        self.consecutive_healthy_borrows = 0

    def reset(self):
        self.funds_available = self._initial_funds_available
        self.consecutive_healthy_borrows = 0

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

    def _supply_withdraw(self, amount: float, plf: PlfPool) -> None:
        """
        Supply (amount > 0) or withdraw (amount < 0) funds to the liquidity pool
        """
        # set default values for user_i_tokens and asset balance if they don't exist
        plf.user_i_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf.asset_name, 0)

        if amount > 0:  # supply
            amount = min(amount, self.funds_available[plf.asset_name])
            # add logging
            logging.info(f"supplying {amount} {plf.asset_name}") if amount > 0 else None
        else:  # withdraw
            withdraw_limit = (
                plf.user_i_tokens[self.name]
                - plf.user_b_tokens[self.name] / plf.collateral_factor
            )
            amount = min(max(amount, -withdraw_limit), 0)
            # add logging
            logging.info(
                f"withdrawing {-amount} {plf.asset_name}"
            ) if amount < 0 else None

        self.funds_available[plf.asset_name] -= amount

        # update liquidity pool
        plf.total_available_funds += amount

        # update i tokens of the user in the pool registry
        plf.user_i_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.interest_token_name] = plf.user_i_tokens[self.name]

        if plf.total_available_funds < 0:
            raise ValueError("total available funds cannot be negative")

    def _borrow_repay(self, amount: float, plf: PlfPool) -> None:
        # set default values for user_b_tokens and funds_available if they don't exist

        plf.user_b_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf.borrow_token_name, 0)
        self.funds_available.setdefault(plf.asset_name, 0)
        self.funds_available.setdefault(plf.interest_token_name, 0)

        if amount > 0:
            # borrow case
            amount = max(
                min(
                    amount,
                    plf.total_available_funds,
                    self.funds_available[plf.interest_token_name]
                    * plf.collateral_factor
                    - self.funds_available[plf.borrow_token_name],
                ),
                0,
            )
            # add logging
            logging.info(f"borrowing {amount} {plf.asset_name}") if amount > 0 else None
        else:
            # repay case
            amount = max(amount, -plf.user_b_tokens[self.name])
            # add logging
            logging.info(f"repaying {-amount} {plf.asset_name}") if amount < 0 else None

        # update liquidity pool
        plf.total_borrowed_funds += amount
        plf.total_available_funds -= amount

        # update b tokens of the user in the pool registry
        plf.user_b_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.borrow_token_name] = plf.user_b_tokens[self.name]

        self.funds_available[plf.asset_name] += amount

        assert (
            plf.total_available_funds >= 0
        ), "total available funds cannot be negative"

    def reactive_action(self, plf: PlfPool) -> None:
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

        max_borrow = existing_supply * plf.collateral_factor / (1 + self.safety_margin)

        if max_borrow > existing_borrow:  # healthy loan
            if plf.borrow_apy <= plf.competing_borrow_apy - 0.01:
                # can borrow up to the buffer
                self._borrow_repay(max_borrow - existing_borrow, plf)
            else:
                # repay as much as you can
                self._borrow_repay(-self.funds_available[plf.asset_name], plf)

            # can deposit all existing funds in the pool if the supply APY is higher than the competing supply APY by 1%
            if plf.supply_apy >= plf.competing_supply_apy + 0.01:
                self._supply_withdraw(self.funds_available[plf.asset_name], plf)
            else:
                # withdraw as much as you can
                self._supply_withdraw(-plf.user_i_tokens[self.name], plf)

            self.consecutive_healthy_borrows += 1
            if self.consecutive_healthy_borrows > 10 and self.safety_margin > 0.05:
                self.safety_margin -= 0.05
        else:  # unhealthy loan
            # repay as much as you can
            self._borrow_repay(-self.funds_available[plf.asset_name], plf)
            existing_borrow = self.funds_available[plf.borrow_token_name]
            existing_supply = self.funds_available[plf.interest_token_name]
            self.consecutive_healthy_borrows = 0

            if existing_borrow > existing_supply * plf.collateral_factor:
                # inject funds to the user to repay the loan
                self.funds_available[plf.asset_name] += existing_borrow * 0.1
                self._borrow_repay(self.funds_available[plf.asset_name], plf)
                self.safety_margin += 0.05


class PlfPool:
    def __init__(
        self,
        env: DefiEnv,
        initiator: User,
        initial_starting_funds: float = 1000,
        collateral_factor: float = 0.85,
        asset_name: str = "dai",
        competing_supply_apy: float = 0.05,
        competing_borrow_apy: float = 0.08,
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
        self.competing_supply_apy = competing_supply_apy
        self.competing_borrow_apy = competing_borrow_apy

        self.env.plf_pools[self.asset_name] = self
        self.reset()

    def reset(self):
        self.previous_reserve: float = 0.0
        self.previous_profit: float = 0.0
        self.reward: float = 0.0

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
        return f"{self.asset_name} PLF: (available funds = {self.total_available_funds}, borrowed funds = {self.total_borrowed_funds}, profit = {self.reserve})"

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
        new_collateral_factor = self.collateral_factor - 0.05
        # Constrain check
        # if the new collateral factor is less than 0
        # if it is out of bounds, then return a very small negative reward and do not update the collateral factor
        if new_collateral_factor < 0:
            self.update_market()
            self.reward = -90
        else:
            self.collateral_factor = new_collateral_factor
            self.update_market()

    def keep_collateral_factor(self) -> None:
        self.update_market()

    def raise_collateral_factor(self) -> None:
        new_collateral_factor = self.collateral_factor + 0.05
        # Constrain check
        # if the new collateral factor is greater than 1
        # if it is out of bounds, then return a very small negative reward and do not update the collateral factor
        if new_collateral_factor > 1:
            self.update_market()
            self.reward = -90
        else:
            self.collateral_factor = new_collateral_factor
            self.update_market()

    def update_market(self) -> None:
        # self.previous_profit = self.profit

        self.accrue_daily_interest()
        for user in self.env.users.values():
            user.reactive_action(self)

        this_profit = self.get_profit()
        if this_profit <= 0:
            self.reward = -10
        else:
            reward_diff = this_profit - self.previous_profit
            self.previous_profit = this_profit
            self.reward = reward_diff
        self.reward = this_profit

    def get_reward(self) -> float:
        """
        get the difference between the profit gained from this episode and the profit gained from the previous episode
        """
        return self.reward

    def get_state(self) -> np.ndarray:
        return np.array(
            [
                self.total_available_funds,
                self.total_borrowed_funds,
                self.collateral_factor,
            ]
        )

    def get_profit(self) -> float:
        previous_reserve = self.previous_reserve
        self.previous_reserve = self.reserve
        return self.reserve - previous_reserve

    @property
    def utilization_ratio(self) -> float:
        ratio = self.total_borrowed_funds / (
            self.total_available_funds + self.total_borrowed_funds
        )
        return max(0, min(1 - 1e-3, ratio))

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
    def reserve(self) -> float:
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
    defi_env = DefiEnv(prices=PriceDict({"tkn": 1}))
    Alice = User(name="alice", env=defi_env, funds_available={"tkn": 2_000})
    plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=500,
        asset_name="tkn",
        collateral_factor=0.8,
    )
    print("Initial")
    print(f"**Alice {Alice} \n")
    print(f"++{plf} \n")

    # Alice react first
    Alice.reactive_action(plf)
    print(f"**Alice {Alice} \n")
    print(f"++{plf} \n")

    defi_env.reset()
    print("After reset")
    print(f"**Alice {Alice} \n")
    print(f"++{plf} \n")

    # then the market update
    plf.raise_collateral_factor()
    plf.update_market()
    Alice.reactive_action(plf)
    print(f"**Alice {Alice} \n")
    print(f"++{plf} \n")
