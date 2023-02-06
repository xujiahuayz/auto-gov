from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from plf_env import constants, settings

# from plf_env.borrow_rates import RATES
from plf_env.computations import (
    compute_compound_interests,
    compute_linear_interests,
    find,
)
from plf_env.constants import (
    ATOKEN_ABS_EPSILON,
    ATOKEN_EPSILON,
    EPSILON,
    INITIAL_TIMESTAMP,
    INTEREST_RATES_PARAMS,
    MIN_BLOCK,
)
from plf_env.core import ProtocolAction, UserAction


@dataclass
class ExternalData:
    prices: Dict[str, float] = field(default_factory=dict)
    volatility: Dict[str, float] = field(default_factory=dict)
    volume: Dict[str, float] = field(default_factory=dict)

    def encode_market_state(self, reserve_address: str) -> np.ndarray:
        return np.array(
            [v[reserve_address] for v in [self.prices, self.volatility, self.volume]]
        )


@dataclass
class MarketUser:
    name: str
    market: Market

    # internal quantity of aTokens, after scaling with index
    _atoken_internal_balance: float = 0.0

    _variable_borrow_internal_balance: float = 0.0

    _cached_stable_borrow: float = 0.0
    unsupplied_amount: float = 0.0

    # whether an asset is used as collateral or not
    collateral_enabled: bool = True

    stable_timestamp: int = 0
    stable_rate: float = 0

    max_atoken_amount: float = 0.0

    # atoken_balance is the balance you see
    @property
    def atoken_balance(self):
        return self._atoken_internal_balance * self.market.supply_index

    @atoken_balance.setter
    def atoken_balance(self, amount: float):
        if settings.PLF_ENV != "test":
            price_in_eth = self.market.external_data.prices[self.market.reserve_address]
            amount_in_eth = price_in_eth * amount
            max_atoken_balance = self.max_atoken_amount * price_in_eth
            assert amount_in_eth >= -max(
                max_atoken_balance * ATOKEN_EPSILON, ATOKEN_ABS_EPSILON
            ), (
                f"internal atoken {self.market.name} balance of {self.name} cannot be negative, "
                f"got {amount} ({amount_in_eth} ETH, max = {max_atoken_balance} ETH)"
            )
        self._atoken_internal_balance = max(amount, 0) / self.market.supply_index
        self.max_atoken_amount = max(self.atoken_balance, self.max_atoken_amount)

    # scaled borrow token, not 1:1 to underlying
    @property
    def variable_borrow(self):
        return (
            self._variable_borrow_internal_balance * self.market.variable_borrow_index
        )

    @variable_borrow.setter
    def variable_borrow(self, amount: float):
        self._variable_borrow_internal_balance = (
            amount / self.market._cached_variable_borrow_index
        )

    @property
    def stable_borrow(self):
        # https://github.com/aave/protocol-v2/blob/7e39178e8f638a85b7bde39a143ab394f6ed0e21/contracts/protocol/tokenization/VariableDebtToken.sol#L75
        # https://github.com/aave/protocol-v2/blob/7e39178e8f638a85b7bde39a143ab394f6ed0e21/contracts/protocol/libraries/logic/ReserveLogic.sol#L85
        cumulated_interests = compute_compound_interests(
            self.stable_rate, self.market.current_timestamp, self.stable_timestamp
        )
        return self._cached_stable_borrow * cumulated_interests

    @stable_borrow.setter
    def stable_borrow(self, amount: float):
        self._cached_stable_borrow = amount
        self.stable_timestamp = self.market.current_timestamp

    @property
    def total_borrow(self) -> float:
        return self.stable_borrow + self.variable_borrow

    def encode_state(self) -> np.ndarray:
        return np.array(
            [
                self.unsupplied_amount,
                self.atoken_balance,
                self.stable_borrow,
                self.variable_borrow,
            ]
        )


@dataclass
class InterestRates:
    supply: float = 0.0
    variable_borrow: float = 0.0
    stable_borrow: float = 0.0


# https://etherscan.io/address/0x16e9c5b85566d1fee99d1e6517183d28cb7c06da#readContract
@dataclass
class InterestRateStrategy:
    variable_borrow_slope_1: float = 0.04
    variable_borrow_slope_2: float = 0.5

    stable_borrow_slope_1: float = 0.16
    stable_borrow_slope_2: float = 0.6

    optimal_utilization_ratio: float = 0.8
    base_variable_borrow_rate: float = 0.01

    def compute_interest_rate(
        self,
        utilization_ratio: float,
        initial_stable_borrow_interest_rate: float,
        overall_borrow_interest_rate: float,
        reserve_factor: float,
    ):
        if utilization_ratio > self.optimal_utilization_ratio:
            excess_utilization_ratio = (
                utilization_ratio - self.optimal_utilization_ratio
            ) / (1 - self.optimal_utilization_ratio)

            stable_borrow_interest_rate = (
                initial_stable_borrow_interest_rate
                + self.stable_borrow_slope_1
                + self.stable_borrow_slope_2 * excess_utilization_ratio
            )
            variable_borrow_interest_rate = (
                self.base_variable_borrow_rate
                + self.variable_borrow_slope_1
                + self.variable_borrow_slope_2 * excess_utilization_ratio
            )
        else:
            actual_to_optimal = utilization_ratio / self.optimal_utilization_ratio
            stable_borrow_interest_rate = (
                initial_stable_borrow_interest_rate
                + self.stable_borrow_slope_1 * actual_to_optimal
            )
            variable_borrow_interest_rate = (
                self.base_variable_borrow_rate
                + actual_to_optimal * self.variable_borrow_slope_1
            )

        supply_interest_rate = (
            overall_borrow_interest_rate * utilization_ratio * (1 - reserve_factor)
        )
        return InterestRates(
            supply=supply_interest_rate,
            stable_borrow=stable_borrow_interest_rate,
            variable_borrow=variable_borrow_interest_rate,
        )

    @classmethod
    def from_params(cls, params: Dict[str, str]):
        return cls(
            variable_borrow_slope_1=float(params["variableRateSlope1"]),
            variable_borrow_slope_2=float(params["variableRateSlope2"]),
            stable_borrow_slope_1=float(params["stableRateSlope1"]),
            stable_borrow_slope_2=float(params["stableRateSlope2"]),
            optimal_utilization_ratio=float(params["OPTIMAL_UTILIZATION_RATE"]),
            base_variable_borrow_rate=float(params["baseVariableBorrowRate"]),
        )


@dataclass(repr=False)
class Market:
    name: str
    reserve_address: str
    external_data: ExternalData = field(default_factory=ExternalData)

    users: Dict[str, MarketUser] = field(default_factory=dict)

    current_timestamp: int = INITIAL_TIMESTAMP
    last_timestamp: int = INITIAL_TIMESTAMP

    interest_rates: InterestRates = field(default_factory=InterestRates)

    average_stable_rate: float = 1.0

    collateral_factor: float = 0.75
    liquidation_threshold: float = 0.8
    liquidation_discount: float = 0.9
    close_factor: float = 0.5
    reserve_factor: float = 0.01

    """`underlying_available` is the actual amount of underlying remaining
    in the contract after lending money to the users
    It corresponds to the balance of the aToken in the underlying ERC contract"""
    underlying_available: float = 0.0

    # scaled one, NOT 1:1 to underlying
    internal_atoken_supply: float = 0.0
    stable_borrow_timestamp: int = 0

    """cached_stable_borrow is the stable borrow with compound interests
    computed until the last time it was updated at `stable_borrow_timestamp`
    The compound interest from `stable_borrow_timestamp` to the current time
    need to be added to compute the total stable borrow
    """
    cached_stable_borrow: float = 0.0

    """`internal_variable_borrow` is the scaled value of the variable borrow
    using the `variable_borrow_index`. The value needs to be divided to
    get the total variable borrows with accrued interests
    """
    internal_variable_borrow: float = 0.0

    _cached_variable_borrow_index: float = 1
    supply_index: float = 1.0

    @property
    def price(self):
        return self.external_data.prices.get(self.reserve_address, 0)

    @cached_property
    def interest_rate_strategy(self):
        params = INTEREST_RATES_PARAMS[self.reserve_address]
        return InterestRateStrategy.from_params(params)

    def get_user(self, user: str):
        return self.users.setdefault(user, MarketUser(user, self))

    def user_atoken_balance(self, user: str):
        return self.get_user(user).atoken_balance

    @cached_property
    def initial_stable_borrow_interest_rate(self):
        return self.interest_rate_strategy.base_variable_borrow_rate

    @property
    def utilization_ratio(self) -> float:
        return self.compute_utilization_ratio()

    def compute_utilization_ratio(self, liquidity_delta: float = 0.0) -> float:
        total_liquidity = (
            self.underlying_available + liquidity_delta + self.total_borrow
        )
        if total_liquidity == 0:
            return 0.0
        return self.total_borrow / total_liquidity

    @property
    def total_borrow(self) -> float:
        """Returns the total stable and variable borrow  with accrued interests up to the current time"""
        return self.stable_borrow + self.variable_borrow

    @property
    def variable_borrow(self) -> float:
        """Returns the total variable borrow with accrued interests up to the current time"""
        return self.internal_variable_borrow * self.variable_borrow_index

    @property
    def stable_borrow(self) -> float:
        """Returns the total stable borrow with accrued interests up to the current time"""
        cumulated_interests = compute_compound_interests(
            self.average_stable_rate,
            self.current_timestamp,
            self.stable_borrow_timestamp,
        )
        return self.cached_stable_borrow * cumulated_interests

    @property
    def variable_borrow_index(self):
        # https://github.com/aave/protocol-v2/blob/7e39178e8f638a85b7bde39a143ab394f6ed0e21/contracts/protocol/libraries/logic/ReserveLogic.sol#L98
        return (
            compute_compound_interests(
                self.interest_rates.variable_borrow,
                self.current_timestamp,
                self.last_timestamp,
            )
            * self._cached_variable_borrow_index
        )

    @property
    def overall_borrow_interest_rate(self) -> float:
        if self.total_borrow == 0:
            return 0.0
        weighted_variable_borrows = (
            self.internal_variable_borrow * self.interest_rates.variable_borrow
        )
        weighted_stable_borrows = (
            self.cached_stable_borrow * self.interest_rates.stable_borrow
        )
        return (weighted_variable_borrows + weighted_stable_borrows) / self.total_borrow

    def __repr__(self):
        return (
            f"Market(name='{self.name}', total_supply={self.underlying_available},"
            f"total_borrow={self.total_borrow})"
        )

    def encode_state(self) -> np.ndarray:
        return np.array(
            [
                self.collateral_factor,
                self.liquidation_threshold,
                self.liquidation_discount,
                self.close_factor,
                self.underlying_available,
                self.total_borrow,
                self.utilization_ratio,
            ]
        )

    def update_interest(self, liquidity_delta: float = 0.0):
        utilization_ratio = self.compute_utilization_ratio(liquidity_delta)
        self.interest_rates = self.interest_rate_strategy.compute_interest_rate(
            utilization_ratio,
            self.initial_stable_borrow_interest_rate,
            self.overall_borrow_interest_rate,
            self.reserve_factor,
        )

    def process_deposit(self, user: str, on_behalf_of: str, amount: float):
        logging.debug(
            "depositing %s %s by %s on behalf of %s, previous balance: %s",
            amount,
            self.name,
            user,
            on_behalf_of,
            self.get_user(on_behalf_of).atoken_balance,
        )
        # https://github.com/aave/protocol-v2/blob/77acec9395c250a0f9df8f73ed99618f9c14a101/contracts/protocol/lendingpool/LendingPool.sol#L104
        # user spends underlying
        self.get_user(user).unsupplied_amount -= amount

        # on_behalf_of receives atokens
        self.get_user(on_behalf_of).atoken_balance += amount
        self.underlying_available += amount
        self.internal_atoken_supply += amount / self.supply_index

        amount_in_eth = self.price * amount
        logging.debug(
            "balance of %s after depositing: %s %s (%s ETH)",
            on_behalf_of,
            amount,
            self.name,
            amount_in_eth,
        )

    def process_redeem(self, user: str, on_behalf_of: str, amount: float):
        # https://github.com/aave/protocol-v2/blob/77acec9395c250a0f9df8f73ed99618f9c14a101/contracts/protocol/lendingpool/LendingPool.sol#L142
        self.get_user(on_behalf_of).unsupplied_amount += amount
        logging.debug(
            "redeeming %s %s by %s on behalf of %s. balance before = %s, supply index = %s, variable borrow index = %s",
            amount,
            self.name,
            user,
            on_behalf_of,
            self.get_user(user).atoken_balance,
            self.supply_index,
            self.variable_borrow_index,
        )
        self.get_user(user).atoken_balance -= amount
        self.underlying_available -= amount
        self.internal_atoken_supply -= amount / self.supply_index
        logging.debug(
            "balance of %s after redeeming: %s",
            user,
            self.get_user(user).atoken_balance,
        )

    def process_borrow(
        self, user: str, on_behalf_of: str, amount: float, is_stable: bool
    ):
        amount_eth = amount * self.price
        logging.debug(
            "borrow %s %s (%s ETH, stable=%s) by %s on behalf of %s, variable borrow index = %s",
            amount,
            self.name,
            amount_eth,
            is_stable,
            user,
            on_behalf_of,
            self.variable_borrow_index,
        )

        if is_stable:
            self.process_borrow_stable(user, on_behalf_of, amount)
        else:
            self.process_borrow_variable(user, on_behalf_of, amount)
        new_underlying_available = self.underlying_available - amount
        # NOTE: can go slightly negative because we are not couting
        # direct transfers to the aToken contract
        self.underlying_available = max(new_underlying_available, 0)

    def process_borrow_stable(self, user: str, on_behalf_of: str, amount: float):
        if amount <= 0:
            return
        total_borrow_stable = self.stable_borrow
        current_stable_rate = self.average_stable_rate
        self.cached_stable_borrow = total_borrow_stable + amount

        self.average_stable_rate = (
            current_stable_rate * total_borrow_stable
            + self.interest_rates.stable_borrow * amount
        ) / self.cached_stable_borrow

        self.stable_borrow_timestamp = self.current_timestamp

        # user receives the borrowed asset without any obligation
        self.get_user(user).unsupplied_amount += amount

        # on_behalf_of's borrow increases
        on_behalf_of_account = self.get_user(on_behalf_of)
        total_stable_borrow = on_behalf_of_account.stable_borrow

        on_behalf_of_account.stable_rate = (
            on_behalf_of_account.stable_rate * on_behalf_of_account.stable_borrow
            + amount * self.interest_rates.stable_borrow
        ) / (on_behalf_of_account.stable_borrow + amount)

        on_behalf_of_account.stable_borrow = amount + total_stable_borrow

    def process_borrow_variable(self, user: str, on_behalf_of: str, amount: float):
        self.internal_variable_borrow += amount / self._cached_variable_borrow_index

        # user gets borrowed money, underlying
        self.get_user(user).unsupplied_amount += amount

        # on_behalf_of's debt increases
        on_behalf_of_account = self.get_user(on_behalf_of)
        on_behalf_of_account.variable_borrow += amount

    def process_repay_variable(self, user: str, on_behalf_of: str, amount: float):
        borrower = self.get_user(on_behalf_of)
        repay_amount = min(borrower.variable_borrow, amount)
        # https://github.com/aave/protocol-v2/blob/7e39178e8f638a85b7bde39a143ab394f6ed0e21/contracts/protocol/tokenization/VariableDebtToken.sol#L124
        self.internal_variable_borrow -= repay_amount

        # user must pay
        self.get_user(user).unsupplied_amount -= repay_amount

        # on_behalf_of's debt decreases
        borrower.variable_borrow -= repay_amount

    def process_repay(
        self, user: str, on_behalf_of: str, amount: float, is_stable: Optional[bool]
    ):
        variable_borrow = self.get_user(on_behalf_of).variable_borrow
        stable_borrow = self.get_user(on_behalf_of).stable_borrow

        amount_eth = amount * self.price
        logging.debug(
            "processing repay of %s %s (%s ETH) for %s on behalf of %s. "
            "stable borrow = %s, variable borrow = %s, is_stable = %s",
            amount,
            self.name,
            amount_eth,
            user,
            on_behalf_of,
            stable_borrow,
            variable_borrow,
            is_stable,
        )

        variable_repay = 0
        if variable_borrow > 0:
            variable_repay = min(amount, variable_borrow)
            self.process_repay_variable(user, on_behalf_of, variable_repay)

        stable_repay = 0
        excess_repay = amount - variable_repay
        if stable_borrow > 0 and excess_repay > 0:
            stable_repay = min(excess_repay, stable_borrow)
            self.process_repay_stable(user, on_behalf_of, stable_repay)

        total_repaid = min(amount, stable_repay + variable_repay)

        self.underlying_available += total_repaid

        borrower = self.get_user(on_behalf_of)
        logging.debug(
            "after repay variable for %s: variable borrow = %s, stable borrow = %s",
            on_behalf_of,
            borrower.variable_borrow,
            borrower.stable_borrow,
        )

    def process_repay_stable(self, user: str, on_behalf_of: str, amount: float):
        borrower = self.get_user(on_behalf_of)
        repay_amount = min(borrower.stable_borrow, amount)
        total_borrow_stable = self.stable_borrow
        if total_borrow_stable <= repay_amount:
            self.average_stable_rate = 0
            self.cached_stable_borrow = 0
        else:
            self.cached_stable_borrow = total_borrow_stable - repay_amount
            scaled_borrows = self.average_stable_rate * total_borrow_stable
            scaled_user_repay = borrower.stable_rate * repay_amount

            # max is only useful if the last user is repaying his borrow
            new_scaled_borrows = max(0, scaled_borrows - scaled_user_repay)
            self.average_stable_rate = new_scaled_borrows / self.cached_stable_borrow

        # user must repay loan
        self.get_user(user).unsupplied_amount -= repay_amount

        # on_behalf_of's loan gets reduced without having to repay
        borrower.stable_borrow -= repay_amount

    def process_swap(self, user: str, was_stable: bool):
        # https://github.com/aave/protocol-v2/blob/master/contracts/protocol/lendingpool/LendingPool.sol#L296-L297
        # TODO: check this balance increase = 0, https://etherscan.io/tx/0x10158f510b83aff48eefff43e0c54c4f5c19ce7f07fbbc8a4ca568d5084b7699#eventlog
        if was_stable:
            amount = self.get_user(user).stable_borrow
            logging.debug(f"stable borrow: {amount}")
            self.process_repay_stable(user, user, amount)
            self.process_borrow_variable(user, user, amount)
        else:
            amount = self.get_user(user).variable_borrow
            logging.debug(f"variable borrow: {amount}")
            self.process_repay_variable(user, user, amount)
            self.process_borrow_stable(user, user, amount)

    # transfer aTokens
    def process_transfer(self, user_sender: str, user_receiver: str, amount: float):
        amount_eth = amount * self.price
        logging.debug(
            "processing transfer of %s %s (%s ETH) from %s to %s",
            amount,
            self.name,
            amount_eth,
            user_sender,
            user_receiver,
        )
        if amount <= 0:
            return
        self.get_user(user_sender).atoken_balance -= amount
        self.get_user(user_receiver).atoken_balance += amount

    # loan market, the market whose asset is used to buy collaterals with a discount
    def liquidate_borrow_market(self, liquidator: str, borrower: str, amount: float):
        eth_amount = amount * self.price
        logging.debug(
            "liquidation repaying %s %s (%s ETH) of %s by %s",
            self.name,
            amount,
            eth_amount,
            borrower,
            liquidator,
        )
        self.underlying_available += amount
        self.internal_atoken_supply += amount / self.supply_index

        self.get_user(liquidator).unsupplied_amount -= amount

        user = self.get_user(borrower)

        variable_amount, stable_amount = amount, 0
        variable_borrows = user.variable_borrow
        if variable_amount > variable_borrows:
            stable_amount = amount - variable_borrows
            variable_amount = variable_borrows

        user.stable_borrow -= stable_amount
        self.cached_stable_borrow -= stable_amount
        user.variable_borrow -= variable_amount
        self.internal_variable_borrow -= variable_amount

    # collateral market, the market whose asset is slashed
    def liquidate_collateral_market(
        self,
        liquidator: str,
        borrower: str,
        amount: float,
        receive_atoken: bool = False,
    ):
        eth_amount = amount * self.price
        logging.debug(
            "liquidating %s %s (%s ETH) of %s by %s (atoken=%s)",
            self.name,
            amount,
            eth_amount,
            borrower,
            liquidator,
            receive_atoken,
        )
        # liquidatee loses collateral
        self.get_user(borrower).atoken_balance -= amount
        # liquidator acquires collateral amount slashed
        if receive_atoken:
            self.get_user(liquidator).atoken_balance += amount
        else:
            # cash out directly, atoken : underlying = 1:1
            self.internal_atoken_supply -= amount / self.supply_index
            self.underlying_available -= amount
            self.get_user(liquidator).unsupplied_amount += amount

    def update_indices(self):
        if self.interest_rates.supply > 0:
            cumulated_supply_interest = compute_linear_interests(
                self.interest_rates.supply, self.current_timestamp, self.last_timestamp
            )
            self.supply_index *= cumulated_supply_interest

            if self.internal_variable_borrow > 0:
                self._cached_variable_borrow_index = self.variable_borrow_index

        self.last_timestamp = self.current_timestamp
        return self.supply_index, self._cached_variable_borrow_index


class Markets:
    def __init__(self, external_data: ExternalData):
        self._markets: Dict[str, Market] = {}
        self.external_data = external_data

    def __getitem__(self, reserve_address: str) -> Market:
        if reserve_address in self._markets:
            return self._markets[reserve_address]
        market = find(lambda m: m["address"] == reserve_address, constants.MARKETS)
        self._markets[reserve_address] = Market(
            market["aTokenSymbol"], reserve_address, external_data=self.external_data
        )
        return self._markets[reserve_address]

    def __setitem__(self, reserve_address: str, market: Market):
        self._markets[reserve_address] = market

    def __iter__(self):
        return iter(self._markets)

    def __len__(self):
        return len(self._markets)

    def __getattr__(self, key):
        if "_markets" not in vars(self):
            raise AttributeError(f"_markets not available")
        return getattr(self._markets, key)


class PLFEnv:
    def __init__(self):
        self.external_data = ExternalData()
        self.markets = Markets(self.external_data)
        self._timestamp = INITIAL_TIMESTAMP
        self.block_number = MIN_BLOCK

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp: int):
        self._timestamp = timestamp
        for market in self.markets.values():
            market.current_timestamp = self.timestamp

    def step(
        self, action: Union[UserAction, ProtocolAction]
    ) -> Tuple[np.ndarray, float]:
        action.execute(self)

        if action.is_user_action():
            assert isinstance(action, UserAction)
            state = self.encode_user_state(action.user)
            reward = self.compute_user_reward(action.user)
        else:
            reward = self.compute_protocol_reward()
            state = self.encode_protocol_state()

        return state, reward

    def reset(self):
        self.external_data = ExternalData()
        self.markets = Markets(self.external_data)
        self.timestamp = 0

    def render(self, mode: str = "human"):
        print(self)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = 42
        return [seed]

    def compute_user_reward(self, user: str) -> float:
        return 0.0

    def compute_protocol_reward(self) -> float:
        return 0.0

    def get_user_supply_value(self, user: str):
        supply_value = 0
        for reserve_address, market in self.markets.items():
            atoken_balance = market.user_atoken_balance(user)
            supply_value += atoken_balance * self.external_data.prices[reserve_address]
        return supply_value

    def get_user_discounted_supply_value_liquidation(
        self, user: str, atoken_deltas: Dict[str, float]
    ) -> float:
        return np.nansum(
            [
                (market.user_atoken_balance(user) - atoken_deltas.get(name, 0))
                * self.external_data.prices[name]
                * market.liquidation_threshold
                for name, market in self.markets.items()
            ]
        )

    def get_user_discounted_supply_value_borrow(self, user: str) -> float:
        return np.nansum(
            [
                market.user_atoken_balance(user)
                * self.external_data.prices[name]
                * market.collateral_factor
                * market.get_user(user).collateral_enabled  # type: ignore
                for name, market in self.markets.items()
            ]
        )

    def get_user_borrow_in_quote(self, user: str) -> float:
        return np.nansum(
            [
                market.get_user(user).total_borrow * self.external_data.prices[name]
                for name, market in self.markets.items()
            ]
        )

    def get_ltv(self, user: str, extra_borrow_in_quote: float = 0.0):
        """
        Get loan (borrow)-to-value (collateral) of a user
        """
        borrow_value = self.get_user_borrow_in_quote(user) + extra_borrow_in_quote
        logging.debug(
            "Borrow value after tx will be %s ============================",
            borrow_value,
        )
        if borrow_value <= 0:
            return 999
        discounted_supply_value = self.get_user_discounted_supply_value_borrow(user)
        logging.debug(
            "Supply value after tx will be %s ============================",
            discounted_supply_value,
        )
        return borrow_value / discounted_supply_value

    def get_health(self, user: str, atoken_deltas: Dict[str, float] = None):
        if atoken_deltas is None:
            atoken_deltas = {}
        borrow_value = self.get_user_borrow_in_quote(user)
        previous_discounted_supply = self.get_user_discounted_supply_value_liquidation(
            user, {}
        )

        # NOTE: to prevent this from failing with deviation due to different
        # prices used. If the borrow is less than 1% of the supply
        # we assume that the position is healthy even if the user wants to
        # withdraw all his funds
        if borrow_value <= EPSILON * previous_discounted_supply:
            return 999
        discounted_supply_value = self.get_user_discounted_supply_value_liquidation(
            user, atoken_deltas
        )
        return discounted_supply_value / borrow_value

    def encode_user_state(self, user: str) -> np.ndarray:
        return np.concatenate(
            [self.encode_market_user_state(market, user) for market in self.markets]
        )

    def encode_market_user_state(self, market: str, user: str) -> np.ndarray:
        return np.concatenate(
            [
                self.markets[market].get_user(user).encode_state(),
                self.encode_market_state(market),
            ]
        )

    def encode_protocol_state(self) -> np.ndarray:
        return np.concatenate(
            [self.encode_market_state(market) for market in self.markets]
        )

    def encode_market_state(self, market: str) -> np.ndarray:
        return np.concatenate(
            [
                self.markets[market].encode_state(),
                self.external_data.encode_market_state(market),
            ]
        )

    def __repr__(self):
        return f"PLFEnv(markets={self.markets}, timestamp={self.timestamp})"
