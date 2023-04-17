from typing import Callable

import numpy as np

from market_env.env import DefiEnv, PlfPool, User
from market_env.utils import PriceDict


def init_env(
    max_steps: int = 30,
    initial_collateral_factor: float = 0.8,
    init_safety_borrow_margin: float = 0.5,
    init_safety_supply_margin: float = 0.5,
    tkn_price_trend_func: Callable[
        [float, int | None], np.ndarray
    ] = lambda x, y: np.ones(x),
    usdc_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x),
    tkn_seed: int | None = None,
    usdc_seed: int | None = None,
) -> DefiEnv:
    defi_env = DefiEnv(
        prices=PriceDict({"tkn": 1, "usdc": 1, "weth": 1}), max_steps=max_steps
    )
    Alice = User(
        name="alice",
        env=defi_env,
        funds_available={"tkn": 20_000, "usdc": 20_000, "weth": 20_000},
        safety_borrow_margin=init_safety_borrow_margin,
        safety_supply_margin=init_safety_supply_margin,
    )
    tkn_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        price_trend_func=tkn_price_trend_func,
        initial_starting_funds=15_000,
        asset_name="tkn",
        collateral_factor=initial_collateral_factor,
        seed=tkn_seed,
    )
    usdc_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        price_trend_func=usdc_price_trend_func,
        initial_starting_funds=15_000,
        asset_name="usdc",
        collateral_factor=initial_collateral_factor,
        seed=usdc_seed,
    )
    weth_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        price_trend_func=lambda x, y: np.ones(x),
        initial_starting_funds=15_000,
        asset_name="weth",
        collateral_factor=initial_collateral_factor,
    )
    return defi_env
