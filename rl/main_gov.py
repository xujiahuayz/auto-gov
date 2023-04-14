import logging
import time
from typing import Callable

import numpy as np
from market_env.caching import cache

from market_env.constants import FIGURES_PATH
from market_env.env import DefiEnv, PlfPool, PriceDict, User
from market_env.utils import generate_price_series
from rl.dqn_gov import Agent
from rl.rl_env import ProtocolEnv
from run_results.plotting import plot_learning_curve


def init_env(
    max_steps: int = 30,
    initial_collateral_factor: float = 0.8,
    init_safety_borrow_margin: float = 0.5,
    init_safety_supply_margin: float = 0.5,
    tkn_price_trend_func: Callable[
        [int, int | None], np.ndarray
    ] = lambda x, y: np.ones(x),
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
        seed=5,
    )
    usdc_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        price_trend_func=lambda x, y: generate_price_series(
            time_steps=x, mu_func=lambda t: 0.01, sigma_func=lambda t: 0.1
        ),
        initial_starting_funds=15_000,
        asset_name="usdc",
        collateral_factor=initial_collateral_factor,
        seed=5,
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


def bench_env(defi_env: DefiEnv) -> list[list[dict]]:
    env = ProtocolEnv(defi_env)
    state_this_game = []
    score = 0
    done = False
    env.reset()
    while not done:
        # get states for plotting
        state_this_game.append(defi_env.state_summary)
        # never change collateral factor
        _, reward, done, _ = env.step(0)
        score += reward
    return state_this_game


# @cache(ttl=60 * 60 * 24 * 7, min_memory_time=0.00001, min_disk_time=0.1)
def train_env(
    defi_env: DefiEnv,
    gamma: float = 0.99,
    n_games: int = 2_000,
    epsilon: float = 1,
    eps_end: float = 0.01,
    eps_dec: float = 5e-5,
    batch_size: int = 128,
    lr: float = 0.003,
    target_net_enabled: bool = False,
) -> tuple[list[float], list[float], list[list[dict]], list[float]]:
    # initialize environment
    env = ProtocolEnv(defi_env)

    # initialize agent
    agent = Agent(
        gamma=gamma,
        epsilon=epsilon,
        batch_size=batch_size,
        n_actions=env.action_space.n,
        eps_end=eps_end,
        input_dims=env.observation_space.shape,
        lr=lr,
        eps_dec=eps_dec,
        target_net_enabled=target_net_enabled,
    )
    # agent = Agent(state_size, action_size)

    scores, eps_history, time_cost, states = [], [], [], []

    for i in range(n_games):
        state_this_game = []
        score = 0
        done = False
        observation = env.reset()
        start_time = time.time()
        while not done:
            # get states for plotting
            state_this_game.append(defi_env.state_summary)
            action = agent.choose_action(observation.astype(np.float32))
            # this checks done or not
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        time_cost.append(time.time() - start_time)
        scores.append(score)
        eps_history.append(agent.epsilon)
        states.append(state_this_game)

        avg_score = np.mean(scores[-30:])
        if i % 50 == 0:
            logging.info(
                "episode: {}, score: {:.2f}, average score: {:.2f}, epsilon: {:.2f}".format(
                    i,
                    score,
                    avg_score,
                    agent.epsilon,
                )
            )
    return scores, eps_history, states, time_cost


def training(
    gamma: float = 0.99,
    n_games: int = 2_000,
    epsilon: float = 1,
    eps_end: float = 0.01,
    eps_dec: float = 5e-5,
    batch_size: int = 128,
    lr: float = 0.003,
    **kwargs,
) -> tuple[list[float], list[float], list[list[dict]], list[float]]:
    defi_env = init_env(**kwargs)
    return train_env(
        defi_env=defi_env,
        gamma=gamma,
        n_games=n_games,
        epsilon=epsilon,
        eps_end=eps_end,
        eps_dec=eps_dec,
        batch_size=batch_size,
        lr=lr,
    )


if __name__ == "__main__":
    # show logging level at info
    logging.basicConfig(level=logging.INFO)
    N_GAMES = 500
    test_env = init_env(
        initial_collateral_factor=0.75,
        max_steps=30,
    )
    (
        training_scores,
        training_eps_history,
        training_collateral_factors,
        _,
    ) = train_env(
        defi_env=test_env,
        n_games=N_GAMES,
        eps_dec=0,
        lr=0.05,
    )
    x = [i + 1 for i in range(N_GAMES)]
    plot_learning_curve(
        x,
        training_scores,
        training_eps_history,
        FIGURES_PATH / "test.pdf",
        "Learning Curve",
    )
