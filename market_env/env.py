# pylint: disable=missing-module-docstring,missing-function-docstring
# make sure User is recognized
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from market_env.constants import (
    DEBT_TOKEN_PREFIX,
    INTEREST_TOKEN_PREFIX,
    PENALTY_REWARD,
)
from market_env.utils import PriceDict, borrow_lend_rates


class DefiEnv:
    def __init__(
        self,
        users: dict[str, User] | None = None,
        prices: PriceDict | None = None,
        plf_pools: dict[str, PlfPool] | None = None,
        max_steps: int = 30,
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
        self.step: int = 0
        self.bad_loan_expenses: float = 0.0
        self.max_steps = max_steps
        self.num_action_pool: int = 3  # lower, keep, raise

    @property
    def prices(self) -> PriceDict:
        return self._prices

    @prices.setter
    def prices(self, value: PriceDict):
        if not isinstance(value, PriceDict):
            raise TypeError("must use PriceDict type")
        self._prices = value

    @property
    def net_position(self) -> float:
        total_reserve = sum(
            (pool.reserve * self.prices[name]) for name, pool in self.plf_pools.items()
        )
        return total_reserve - self.bad_loan_expenses

    @property
    def state_summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "net_position": self.net_position,
            "pools": {
                name: {
                    "collateral_factor": p.collateral_factor,
                    "reserve": p.reserve,
                    "borrow_apy": p.borrow_apy,
                    "supply_apy": p.supply_apy,
                    "utilization_ratio": p.utilization_ratio,
                    "price": p.env.prices[name],
                }
                for name, p in self.plf_pools.items()
            },
        }

    def __repr__(self) -> str:
        return f"DefiEnv({self.state_summary})"

    def update_collateral_factor(self, action: int) -> None:
        num_pools: int = len(self.plf_pools)
        if not 0 <= action < self.num_action_pool**num_pools:
            raise ValueError("action must be between 0 and {num_pools**3 -1}")

        for i, plf in enumerate(self.plf_pools.values()):
            exponent = num_pools - i
            this_action = (
                action
                % (self.num_action_pool**exponent)
                // (self.num_action_pool ** (exponent - 1))
            )
            match this_action:
                case 0:
                    plf.keep_collateral_factor()
                case 1:
                    plf.lower_collateral_factor()
                case 2:
                    plf.raise_collateral_factor()

        for user in self.users.values():
            user.reactive_action()

    def get_reward(self) -> float:
        if self.net_position < 0:
            self.step = self.max_steps  # end the episode
            return PENALTY_REWARD * len(self.plf_pools)
        return sum(
            pool.reward
            if pool.reward == PENALTY_REWARD
            else pool.get_profit() * self.prices[name]
            for name, pool in self.plf_pools.items()
        )

    def get_state(self) -> np.ndarray:
        return np.concatenate(self._apply_to_all_pools(PlfPool.get_state))

    def is_done(self) -> bool:
        """
        Returns True if step reaches max_steps, otherwise False
        """
        self.step += 1
        self.bad_loan_expenses = 0  # reset bad loan expenses
        return self.step >= self.max_steps

    def reset(self) -> None:
        self.step = 0

        # reset bad loan expenses, shouldn't be necessary but just in case
        self.bad_loan_expenses = 0
        for user in self.users.values():
            user.reset()
        self._apply_to_all_pools(PlfPool.reset)

    def _apply_to_all_pools(self, func) -> list:
        return [func(pool) for pool in self.plf_pools.values()]


class User:
    def __init__(
        self,
        env: DefiEnv,
        name: str,
        safety_borrow_margin=0.05,
        safety_supply_margin=0.05,
        funds_available: dict[str, float] | None = None,
    ):
        assert name not in env.users, f"User {name} exists"
        self.env = env
        # add the user to the environment
        self.env.users[name] = self

        if funds_available is None:
            funds_available = {"dai": 0.0, "eth": 0.0}
        self._initial_funds_available = funds_available.copy()

        self.name = name
        self._initial_safety_borrow_margin = safety_borrow_margin
        self._initial_safety_supply_margin = safety_supply_margin
        self.reset()

    def reset(self):
        self.funds_available = self._initial_funds_available.copy()
        self.consecutive_good_borrows = 0
        self.consecutive_good_supplies = 0
        self.safety_borrow_buffer = self._initial_safety_borrow_margin
        self.safety_supply_buffer = self._initial_safety_supply_margin

    @property
    def wealth(self) -> float:
        user_wealth = sum(
            value * self.env.prices[asset_name]
            for asset_name, value in self.funds_available.items()
        )
        logging.debug(f"{self.name}'s wealth in USD: {user_wealth}")

        return user_wealth

    @property
    def existing_borrow_value(self) -> float:
        return sum(
            self.funds_available[plf.borrow_token_name]
            * self.env.prices[plf.asset_name]
            for plf in self.env.plf_pools.values()
        )

    @property
    def existing_supply_value(self) -> float:
        return sum(
            self.funds_available[plf.interest_token_name] * self.env.prices[name]
            for name, plf in self.env.plf_pools.items()
        )

    @property
    def max_borrowable_value(self) -> float:
        return sum(
            self.funds_available[plf.interest_token_name]
            * self.env.prices[plf.asset_name]
            * plf.collateral_factor
            for plf in self.env.plf_pools.values()
        )

    @property
    def total_asset_value(self) -> float:
        """
        Total asset value in USD
        including interest-bearing tokens and not-yet-supplied assets
        not including borrow tokens
        """
        return sum(
            (
                self.funds_available[plf.interest_token_name]
                + self.funds_available[plf.asset_name]
            )
            * self.env.prices[plf.asset_name]
            for plf in self.env.plf_pools.values()
        )

    def __repr__(self) -> str:
        return f"name: {self.name}, funds available: {self.funds_available}, initial funds: {self._initial_funds_available}, wealth: {self.wealth}"

    def _supply_withdraw(self, amount: float, plf: PlfPool) -> None:
        """
        Supply (amount > 0) or withdraw (amount < 0) funds to the liquidity pool
        """
        # set default values for user_i_tokens and asset balance if they don't exist
        plf.user_i_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf.asset_name, 0)

        assert (
            self.funds_available[plf.interest_token_name] >= 0
        ), "user cannot have negative interest-bearing asset balance"

        if amount >= 0:  # supply
            # will never supply EVERYTHING - always leave some safety margin
            amount = min(
                amount,
                self.funds_available[plf.asset_name] / (1 + self.safety_supply_buffer),
            )
            if 0 <= amount < 1e-9:
                return
            log_text = f"supplying {amount} {plf.asset_name}"

        else:  # withdraw
            supported_borrow_without_this_plf = sum(
                self.funds_available[w.interest_token_name]
                * w.collateral_factor
                * self.env.prices[w.asset_name]
                for w in self.env.plf_pools.values()
                if w is not plf
            )

            minimum_supply = max(
                0,
                (
                    (self.existing_borrow_value - supported_borrow_without_this_plf)
                    / plf.collateral_factor
                    / self.env.prices[plf.asset_name]
                ),
            )

            withdraw_limit = (
                self.funds_available[plf.interest_token_name] - minimum_supply
            )
            if withdraw_limit <= 0:
                # handle some funky rounding errors
                return
            log_text = f"withdrawing {-amount} {plf.asset_name} when pool has {plf.total_available_funds} {plf.asset_name} at limit {withdraw_limit}"
            amount = max(amount, -withdraw_limit, -plf.total_available_funds)

        logging.debug(log_text)

        self.funds_available[plf.asset_name] -= amount

        # update liquidity pool
        plf.total_available_funds += amount

        # update i tokens of the user in the pool registry
        plf.user_i_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.interest_token_name] = plf.user_i_tokens[self.name]

        assert (
            self.funds_available[plf.asset_name] >= 0
        ), "user cannot have negative asset balance but has %s of %s" % (
            self.funds_available[plf.asset_name],
            plf.asset_name,
        )

        if plf.total_available_funds < 0:
            raise ValueError("total available funds cannot be negative at \n %s" % plf)

    def _borrow_repay(self, amount: float, plf: PlfPool) -> None:
        # set default values for user_b_tokens and funds_available if they don't exist

        plf.user_b_tokens.setdefault(self.name, 0)
        self.funds_available.setdefault(plf.borrow_token_name, 0)
        self.funds_available.setdefault(plf.asset_name, 0)
        self.funds_available.setdefault(plf.interest_token_name, 0)

        if amount >= 0:
            # borrow case
            # will never borrow EVERYTHING - always leave some safety margin
            additional_borrowable_amount = (
                self.max_borrowable_value
                - self.existing_borrow_value * (1 + self.safety_borrow_buffer)
            ) / self.env.prices[plf.asset_name]
            amount = max(
                min(
                    amount,
                    plf.total_available_funds,
                    additional_borrowable_amount,
                ),
                0,
            )
            if 0 <= amount < 1e-9:  # if amount is too small,
                return
        else:
            # repay case
            amount = max(
                amount,
                -plf.user_b_tokens[self.name],
                -self.funds_available[plf.asset_name],
            )

        logging.debug(
            f"borrowing {amount} {plf.borrow_token_name}"
            if amount > 0
            else f"repaying {-amount} {plf.borrow_token_name}"
        )
        # update liquidity pool
        plf.total_borrowed_funds += amount
        plf.total_available_funds -= amount

        # update b tokens of the user in the pool registry
        plf.user_b_tokens[self.name] += amount

        # matching balance in user's account to pool registry record
        self.funds_available[plf.borrow_token_name] = plf.user_b_tokens[self.name]

        self.funds_available[plf.asset_name] += amount

        assert plf.total_available_funds >= 0, (
            "total available funds cannot be negative at \n %s" % plf
        )

    def reactive_action(self) -> None:
        """
        supply funds to the liquidity pool in response to market conditions
        """
        for name, plf in self.env.plf_pools.items():
            assert self.env.prices[name] == plf.asset_price_history[self.env.step]
        if self.existing_borrow_value < self.max_borrowable_value:  # healthy loan
            for plf in self.env.plf_pools.values():
                borrow_apy_advantage = plf.competing_borrow_apy + 0.01 - plf.borrow_apy
                if borrow_apy_advantage > 0:
                    self.consecutive_good_borrows += 1
                    # the bigger advantage, the higher the borrow amount
                    self._borrow_repay(
                        plf.total_available_funds * (1 - np.exp(-borrow_apy_advantage)),
                        plf,
                    )
                else:
                    # the bigger |borrow_apy_advantage|, the more you repay
                    self._borrow_repay(
                        -self.funds_available[plf.asset_name]
                        * (1 - np.exp(borrow_apy_advantage)),
                        plf,
                    )

                # set APY thresholds -- below withdraw, above supply
                supply_apy_advantage = plf.supply_apy - plf.competing_supply_apy + 0.01
                if supply_apy_advantage < 0:
                    self.consecutive_good_supplies = 0
                    # withdraw, the bigger |supply_apy_margin|, the more you withdraw
                    self._supply_withdraw(
                        -plf.user_i_tokens[self.name]
                        * (1 - np.exp(supply_apy_advantage)),
                        plf,
                    )
                else:
                    self.consecutive_good_supplies += 1
                    # supply as much as you can
                    self._supply_withdraw(
                        self.funds_available[plf.asset_name]
                        * (1 - np.exp(-supply_apy_advantage)),
                        plf,
                    )

                # if plf.total_b_tokens != 0:
                #     assert (
                #         plf.total_i_tokens > 0
                #     ), f"total i tokens is {plf.total_i_tokens} when total b tokens is {plf.total_b_tokens}"

        else:  # unhealthy loan
            self.consecutive_good_borrows = 0
            if self.existing_borrow_value >= self.existing_supply_value > 0:
                # not worth repaying the loan, prefer defaulting
                print("USER IS DEFAULTING!!! WRITING OFF LOAN")
                self.env.bad_loan_expenses += self.existing_borrow_value
            else:
                # start repaying from the one with the highest borrow APY
                # sort the pools by borrow APY
                sorted_plf_pools = sorted(
                    self.env.plf_pools.values(),
                    key=lambda x: x.borrow_apy,
                    reverse=True,
                )

                while (self.existing_borrow_value > self.max_borrowable_value) and (
                    sorted_plf_pools
                ):
                    plf = sorted_plf_pools.pop(0)
                    self._borrow_repay(
                        -(self.existing_borrow_value - self.max_borrowable_value)
                        * 1.1
                        / plf.env.prices[plf.asset_name],
                        plf,
                    )

                # second round: if repaying does not suffice, inject capital to repay more, note that this will hurt the user's confidence
                sorted_plf_pools = sorted(
                    self.env.plf_pools.values(),
                    key=lambda x: x.borrow_apy,
                    reverse=True,
                )
                while (
                    self.existing_borrow_value > self.max_borrowable_value
                ) and sorted_plf_pools:
                    self.safety_borrow_buffer += 0.1
                    # inject funds to the user to repay the loan
                    plf = sorted_plf_pools.pop(0)
                    self.funds_available[plf.asset_name] += min(
                        (self.existing_borrow_value - self.max_borrowable_value)
                        * 1.1
                        / plf.env.prices[plf.asset_name],
                        self.funds_available[plf.borrow_token_name],
                    )
                    self._borrow_repay(-self.funds_available[plf.asset_name], plf)
                    # if plf.total_b_tokens != 0:
                    #     assert (
                    #         plf.total_i_tokens > 0
                    #     ), f"total i tokens is {plf.total_i_tokens} when total b tokens is {plf.total_b_tokens}"

        if self.consecutive_good_borrows > 20 and self.safety_borrow_buffer > 0.05:
            self.safety_borrow_buffer -= 0.05
        if self.consecutive_good_supplies > 20 and self.safety_supply_buffer > 0.05:
            self.safety_supply_buffer -= 0.05


class PlfPool:
    def __init__(
        self,
        env: DefiEnv,
        initiator: User,
        price_trend_func: Callable[
            [int, int | None], np.ndarray
        ] = lambda x, y: np.ones(x),
        initial_starting_funds: float = 1000,
        collateral_factor: float = 0.85,
        asset_name: str = "dai",
        competing_supply_apy: float = 0.05,
        competing_borrow_apy: float = 0.15,
        seed: int | None = 0,
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
        self.asset_name = asset_name
        self.competing_supply_apy = competing_supply_apy
        self.competing_borrow_apy = competing_borrow_apy
        self._initial_collar_factor = collateral_factor
        self._initial_asset_price = self.env.prices[self.asset_name]
        self.seed = seed
        # deterministically generate price history
        self.price_trend_func = price_trend_func

        self.reset()

    def reset(self):
        self.env.plf_pools[self.asset_name] = self
        self.asset_price_history = self._initial_asset_price * self.price_trend_func(
            self.env.max_steps, self.seed
        )
        self.user_i_tokens: dict[str, float] = {
            self.initiator.name: self.initial_starting_funds
        }
        self.user_b_tokens: dict[str, float] = {self.initiator.name: 0.0}
        self._collateral_factor: float = self._initial_collar_factor
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

        # add interest-bearing token into initiator's wallet
        self.initiator.funds_available[
            self.interest_token_name
        ] = self.initial_starting_funds
        self.initiator.funds_available[self.borrow_token_name] = 0
        self.env.prices[self.asset_name] = self._initial_asset_price

    def __repr__(self) -> str:
        return (
            f"{self.asset_name} PLF: "
            + f"Collateral factor: {self.collateral_factor:.2f} \n"
            + f"Supply APY: {self.supply_apy:.5f} \n"
            + f"Borrow APY: {self.borrow_apy:.5f} \n"
            + f"total borrow: {self.total_b_tokens:.4f} \n"
            + f"total supply: {self.total_i_tokens:.4f} \n"
            + f"total available funds: {self.total_available_funds:.4f} \n"
            + f"utilization rate: {self.utilization_ratio:.5f}\n"
            + f"reserve: {self.reserve:.4f} \n"
        )

    @property
    def collateral_factor(self):
        return self._collateral_factor

    @collateral_factor.setter
    def collateral_factor(self, value: float):
        if value < 0 or value > 1:
            raise ValueError("collateral factor must be between 0 and 1")
        self._collateral_factor = value

    def update_asset_price(self) -> None:
        new_price = self.asset_price_history[self.env.step]
        assert new_price > 0, "asset price cannot be negative"
        self.env.prices[self.asset_name] = new_price

    # actions
    def lower_collateral_factor(self) -> None:
        # affect users who are borrowing from this pool
        new_collateral_factor = self.collateral_factor - 0.025
        # Constrain check
        # if the new collateral factor is less than 0
        # if it is out of bounds, then return a very small negative reward and do not update the collateral factor
        if new_collateral_factor < 0:
            self.update_market()
            self.reward = PENALTY_REWARD
        else:
            self.collateral_factor = new_collateral_factor
            for user in self.env.users.values():
                user.safety_borrow_buffer += 0.05
            self.update_market()

    def keep_collateral_factor(self) -> None:
        self.update_market()

    def raise_collateral_factor(self) -> None:
        new_collateral_factor = self.collateral_factor + 0.025
        # Constrain check
        # if the new collateral factor is greater than 1
        # if it is out of bounds, then return a very small negative reward and do not update the collateral factor
        if new_collateral_factor > 1:
            self.update_market()
            self.reward = PENALTY_REWARD
        else:
            self.collateral_factor = new_collateral_factor
            # affect users who are supplying to this pool with higher exposure to default risk
            for user in self.env.users.values():
                user.safety_supply_buffer += 0.05
            self.update_market()

    def update_market(self) -> None:
        self.accrue_daily_interest()
        self.update_asset_price()
        self.reward = 0  # reset reward

    # def get_reward(self) -> float:
    #     """
    #     get the difference between the profit gained from this episode and the profit gained from the previous episode
    #     """
    #     # self.reward = self.get_profit()
    #     return self.reward

    def get_state(self) -> np.ndarray:
        return np.array(
            [
                self.total_available_funds,
                self.total_borrowed_funds,
                self.total_i_tokens,
                self.total_b_tokens,
                self.collateral_factor,
                self.utilization_ratio,
                self.supply_apy,
                self.borrow_apy,
                self.env.prices[self.asset_name],
            ]
        )

    def get_profit(self) -> float:
        previous_reserve = self.previous_reserve
        self.previous_reserve = self.reserve
        return self.reserve - previous_reserve

    @property
    def utilization_ratio(self) -> float:
        if self.total_i_tokens == 0:
            return 0
        return self.total_b_tokens / self.total_i_tokens

    # max(0, min(1 - 1e-3, ratio))

    @property
    def supply_apy(self) -> float:
        _, rs = borrow_lend_rates(util_rate=self.utilization_ratio)
        return rs

    @property
    def borrow_apy(self) -> float:
        rb, _ = borrow_lend_rates(util_rate=self.utilization_ratio)
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
        spread: float = 0.2,
        rb_factor: float = 20,
    ) -> tuple[float, float]:
        # TODO: check where to put the factors
        """
        calculate borrow and supply rates based on utilization ratio
        with an arbitrarily-set shape
        """
        # theoretically unnecessary, but to avoid floating point errors
        if util_rate == 0:
            return 0, 0

        assert (
            -1e-9 < util_rate
        ), f"utilization ratio must be non-negative, but got {util_rate}"
        constrained_util_rate = max(0, min(util_rate, 0.97))

        borrow_rate = constrained_util_rate / (rb_factor * (1 - constrained_util_rate))
        daily_borrow_interest = (1 + borrow_rate) ** (1 / 365) - 1
        daily_supply_interest = daily_borrow_interest * constrained_util_rate
        supply_rate = ((1 + daily_supply_interest) ** 365 - 1) * (1 - spread)

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
    logging.basicConfig(level=logging.INFO)
    # https://docs.aave.com/risk/v/aave-v2/asset-risk/risk-parameters
    # initialize environment
    defi_env = DefiEnv(prices=PriceDict({"tkn": 3, "usdc": 0.1, "weth": 1}))
    Alice = User(
        name="alice",
        env=defi_env,
        funds_available={"tkn": 10_000, "usdc": 200_000, "weth": 20_000},
    )
    tkn_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=5_000,
        asset_name="tkn",
        collateral_factor=0.8,
    )
    usdc_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=150_000,
        asset_name="usdc",
        collateral_factor=0.8,
    )
    weth_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=15_000,
        asset_name="weth",
        collateral_factor=0.8,
    )
    print("Initial =============== \n")
    print(Alice)
    print(tkn_plf)
    print(usdc_plf)
    print(weth_plf)

    # Alice react first

    Alice.reactive_action()
    print(Alice)
    print(tkn_plf)
    print(usdc_plf)
    print(weth_plf)

    defi_env.reset()
    print("After reset")
    print(Alice)
    print(tkn_plf)
    print(usdc_plf)
    print(weth_plf)

    # then the market update
    tkn_plf.update_market()
    usdc_plf.raise_collateral_factor()
    Alice.reactive_action()
    print(Alice)
    print(tkn_plf)
    print(usdc_plf)
    print(weth_plf)
