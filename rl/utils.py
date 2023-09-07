from typing import Callable

import numpy as np
import pickle

from market_env.env import DefiEnv, PlfPool, User
from market_env.utils import PriceDict
from market_env.constants import DATA_PATH


def init_env(
    max_steps: int = 30,
    attack_steps: list[int] | None = None,
    initial_collateral_factor: float = 0.8,
    init_safety_borrow_margin: float = 0.5,
    init_safety_supply_margin: float = 0.5,
    tkn_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x + 1),
    usdc_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x + 1),
    tkn_seed: int | None = None,
    usdc_seed: int | None = None,
) -> DefiEnv:
    defi_env = DefiEnv(
        prices=PriceDict({"tkn": 1, "usdc": 1, "weth": 1}),
        max_steps=max_steps,
        attack_steps=attack_steps,
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
        competing_collateral_factor=0,
    )
    usdc_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        price_trend_func=usdc_price_trend_func,
        initial_starting_funds=15_000,
        asset_name="usdc",
        collateral_factor=initial_collateral_factor,
        seed=usdc_seed,
        competing_collateral_factor=0.65,
    )
    weth_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        price_trend_func=lambda x, y: np.ones(x),
        initial_starting_funds=15_000,
        asset_name="weth",
        collateral_factor=initial_collateral_factor,
        competing_collateral_factor=0.7,
    )
    return defi_env


def save_the_nth_model(num, prefix_name, models):
    """
    num is the num-th model to be saved
    prefix_name is the prefix of the file name (prefix_name + num + ".pkl")
    models is the list of models
    """
    with open(str(DATA_PATH / prefix_name) + str(num) + ".pkl", "wb") as f:
        pickle.dump(models[num], f)


def load_saved_model(num, prefix_name):
    """
    num is the num-th model to be loaded
    prefix_name is the prefix of the file name (prefix_name + num + ".pkl")
    """
    with open(str(DATA_PATH / prefix_name) + str(num) + ".pkl", "rb") as f:
        return pickle.load(f)