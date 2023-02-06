from dataclasses import field
import logging
from typing import Optional
from attr import validate

import numpy as np

from plf_env.actions import (
    UpdateCollateralFactor,
    UpdateLiquidationDiscount,
    UpdateLiquidationThreshold,
    UpdateReserveFactor,
)
from plf_env.constants import ADDRESSES, ADDRESSES_BY_NAME
from plf_env.core import ProtocolAction
from plf_env.env import PLFEnv


dai, weth, usdt = (
    ADDRESSES_BY_NAME["DAI"],
    ADDRESSES_BY_NAME["WETH"],
    ADDRESSES_BY_NAME["USDT"],
)


class Strategist:
    def __init__(self, plf_env: PLFEnv, address: str = "strategist"):
        self.plf_env = plf_env
        self.action_history: dict[int, list[ProtocolAction]] = {}
        self.wealths: dict[int, float] = {}
        self.address = address

    @property
    def net_asset(self) -> float:

        return np.nansum(
            [
                (
                    self.plf_env.markets[m].internal_atoken_supply
                    * self.plf_env.markets[m].supply_index
                    - self.plf_env.markets[m].total_borrow
                )
                * self.plf_env.external_data.prices[m]
                for m in self.plf_env.markets
            ]
        )

    def generate_actions(self) -> list[ProtocolAction]:
        actions = self._generate_actions()
        if actions:
            self.action_history[self.plf_env.block_number] = actions
            logging.info(f"strategist action recorded in history")
        self.wealths[self.plf_env.block_number] = self.net_asset
        return actions

    def _generate_actions(self) -> list[ProtocolAction]:
        raise NotImplementedError()


class NaiveStrategist(Strategist):
    def _generate_actions(self) -> list[ProtocolAction]:

        if len(self.action_history) > 0:
            # only allow one action at the beginning
            return []

        actions = [
            UpdateCollateralFactor(market=dai, value=0.8),
            UpdateLiquidationDiscount(market=weth, value=0.99),
        ]
        self.action_history[self.plf_env.block_number] = actions
        return actions
