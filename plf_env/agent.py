from dataclasses import field
import logging
from typing import Optional

import numpy as np

from plf_env.actions import Borrow, Deposit, Exchange, Redeem, Repay
from plf_env.constants import ADDRESSES, ADDRESSES_BY_NAME
from plf_env.core import UserAction
from plf_env.env import PLFEnv


dai, weth, usdt = (
    ADDRESSES_BY_NAME["DAI"],
    ADDRESSES_BY_NAME["WETH"],
    ADDRESSES_BY_NAME["USDT"],
)


class Agent:
    def __init__(
        self,
        address: str,
        plf_env: PLFEnv,
        initial_unsupplied_amounts: Optional[dict[str, float]] = field(
            default_factory=dict
        ),
    ):
        self.address = address
        self.plf_env = plf_env
        self.action_history: dict[int, list[UserAction]] = {}
        self.wealths: dict[int, float] = {}

        # record initial quantity to each market
        # for w in ADDRESSES:
        #     if w not in initial_unsupplied_amounts:
        #         initial_unsupplied_amounts[w] = 0
        #     plf_env.markets[w].get_user(
        #         self.address
        #     ).unsupplied_amount = initial_unsupplied_amounts[w]

        if initial_unsupplied_amounts is None:
            initial_unsupplied_amounts = {w: 0 for w in ADDRESSES}
        for market, quantity in initial_unsupplied_amounts.items():
            plf_env.markets[market].get_user(self.address).unsupplied_amount = quantity
        self.inital_wealth = self.wealth

    @property
    def wealth(self) -> float:
        market_users = [
            self.plf_env.markets[m].get_user(self.address) for m in ADDRESSES
        ]

        # TODO: the if check shouldn't be needed -- may need to start simulation a bit later when oracle prices are available
        return np.nansum(
            [
                (
                    market_user.unsupplied_amount
                    + market_user.atoken_balance
                    - market_user.total_borrow
                )
                * self.plf_env.external_data.prices[market_user.market.reserve_address]
                for market_user in market_users
                if market_user.market.reserve_address
                in self.plf_env.external_data.prices
            ]
        )

    # @property
    # def return_on_wealth(self) -> float:
    #     return self.wealth / self.inital_wealth - 1

    # used to add initial funds to the agent
    def update_asset_balance_externally(self, market: str, quantity: float):
        market_user = self.plf_env.markets[market].get_user(self.address)

        # quantity can be negative
        assert (
            market_user.unsupplied_amount + quantity >= 0
        ), "unsupplied amount must be non-genative"

        # update balance
        market_user.unsupplied_amount += quantity

    def generate_actions(self) -> list[UserAction]:
        actions = self._generate_actions()
        if actions:
            self.action_history[self.plf_env.block_number] = actions
            logging.info(f"{self.address} action recorded in history")
        self.wealths[self.plf_env.block_number] = self.wealth
        return actions

    def _generate_actions(self) -> list[UserAction]:
        raise NotImplementedError()


class MarketAggregateAgent(Agent):
    def _generate_actions(self) -> list[UserAction]:
        dai_market = self.plf_env.markets[dai]
        usdt_market = self.plf_env.markets[usdt]
        weth_market = self.plf_env.markets[weth]
        return [
            Deposit(
                user=self.address,
                on_behalf_of=self.address,
                market=dai,
                value=1e4,
            ),
            Borrow(
                user=self.address,
                on_behalf_of=self.address,
                market=dai,
                value=lambda: (
                    dai_market.get_user(self.address).atoken_balance
                    * dai_market.collateral_factor
                    - dai_market.get_user(self.address).total_borrow
                )
                * 0.7,
            ),
            Deposit(
                user=self.address,
                on_behalf_of=self.address,
                market=weth,
                value=1e2,
            ),
            Borrow(
                user=self.address,
                on_behalf_of=self.address,
                market=weth,
                value=lambda: (
                    weth_market.get_user(self.address).atoken_balance
                    * weth_market.collateral_factor
                    - weth_market.get_user(self.address).total_borrow
                )
                * 0.7,
            ),
            Deposit(
                user=self.address,
                on_behalf_of=self.address,
                market=usdt,
                value=1e4,
            ),
            Borrow(
                user=self.address,
                on_behalf_of=self.address,
                market=usdt,
                value=lambda: (
                    usdt_market.get_user(self.address).atoken_balance
                    * usdt_market.collateral_factor
                    - usdt_market.get_user(self.address).total_borrow
                )
                * 0.7,
            ),
        ]
        pass


class NaiveAgent(Agent):
    """
    only deposits once at the very beginning
    """

    def _generate_actions(self) -> list[UserAction]:
        logging.debug(self.action_history)
        usdt_balance = self.plf_env.markets[ADDRESSES_BY_NAME["USDT"]].get_user(
            self.address
        )
        logging.debug(
            f"""
        {self.address}'s total wealth: {self.wealth}
        _internal_atoken: {usdt_balance._atoken_internal_balance}
        atoken: {usdt_balance.atoken_balance}
        unsupplied: {usdt_balance.unsupplied_amount}
        """
        )
        if len(self.action_history) > 0:
            # only allow one action at the beginning
            return []

        return [
            Deposit(
                user=self.address,
                on_behalf_of=self.address,
                market=ADDRESSES_BY_NAME["DAI"],
                value=9.999,
            )
        ]  # type: list[UserAction]


class SpiralBorrowAgent(Agent):
    """
    Deposit ETH -> borrow DAI -> swap DAI to ETH -> wait until ETH goes up
    -> swap ETH to DAI -> repay DAI
    and so on and so forth
    """

    def _generate_actions(self) -> list[UserAction]:
        dai_market = self.plf_env.markets[dai]
        user_dai_balance = dai_market.get_user(self.address)
        weth_market = self.plf_env.markets[weth]
        user_weth_balance = weth_market.get_user(self.address)
        prices = self.plf_env.external_data.prices

        if not self.action_history:
            # print(self.action_history)
            # action.append(

            # )  # supply all the
            # print(action)
            action = [
                Deposit(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=weth,
                    value=user_weth_balance.unsupplied_amount,
                )
            ]  # type: list[UserAction]

            # logging.info(
            #     f"action generated: {self.address} deposit {weth_market.name} of %s",
            #     user_weth_balance.unsupplied_amount,
            # )
        elif prices[dai] > 1 / 700 and (
            (  # borrow 80% dai that you can maximally buy
                borrowable := min(
                    (
                        user_weth_balance.atoken_balance
                        * prices[weth]
                        / prices[dai]
                        * dai_market.collateral_factor
                        * 0.8
                    )
                    - user_dai_balance.total_borrow,
                    dai_market.underlying_available,
                )
            )
            > 1e-4
        ):

            action = [
                Borrow(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=dai,
                    value=borrowable,
                ),
                Exchange(
                    user=self.address,
                    from_market=dai,
                    to_market=weth,
                    value=lambda: user_dai_balance.unsupplied_amount,
                ),  # exchange all dai to weth
                Deposit(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=weth,
                    value=lambda: user_weth_balance.unsupplied_amount,
                ),  # deposit all eth
            ]
            logging.info(f"action generated: {self.address} borrwo-exchange-deposit")
        elif prices[dai] < 1 / 1200 and user_dai_balance.total_borrow > 1e-4:
            action = [
                Redeem(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=weth,
                    value=user_dai_balance.total_borrow
                    * prices[dai]
                    / prices[weth]
                    * 1.2,
                ),
                Exchange(
                    user=self.address,
                    from_market=weth,
                    to_market=dai,
                    value=lambda: user_weth_balance.unsupplied_amount,
                ),  # exchange all weth back to dai
                Repay(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=dai,
                    value=lambda: user_dai_balance.unsupplied_amount,
                ),  # withdraw some eth to repay everything, excess funds should be returned anyway
            ]
            logging.info(f"{self.address} performed exchange-repay")
        else:
            action = []
        return action


class SpeculativeAgent(Agent):
    """
    Deposit ETH -> borrow DAI -> swap DAI to ETH -> wait until ETH goes up
    -> swap ETH to DAI -> repay DAI
    """

    def _generate_actions(self) -> list[UserAction]:
        dai, weth = ADDRESSES_BY_NAME["DAI"], ADDRESSES_BY_NAME["WETH"]

        if self.plf_env.external_data.prices[dai] > 1 / 800 and not self.action_history:
            """
            do deposit + borrow + swap when DAI price (in ETH) is high
            """
            action = [
                Deposit(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=weth,
                    value=9.999,
                ),
                Borrow(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=dai,
                    value=0.001,
                ),
                Exchange(
                    user=self.address,
                    from_market=dai,
                    to_market=weth,
                    value=0.001,
                ),
            ]
            logging.info(f"action generated: {self.address} deposit-borrow-exchange")
        elif (
            self.plf_env.external_data.prices[dai] < 1 / 1200
            and len(self.action_history) == 1
        ):
            action = [
                Exchange(
                    user=self.address,
                    from_market=weth,
                    to_market=dai,
                    value=0.1,
                ),
                # repay everything, excess funds should be returned anyway
                Repay(
                    user=self.address,
                    on_behalf_of=self.address,
                    market=dai,
                    value=self.plf_env.markets[dai]
                    .get_user(self.address)
                    .unsupplied_amount,
                ),
            ]
            logging.info(f"action generated: {self.address} performed exchange-repay")
        else:
            action = []
        return action
