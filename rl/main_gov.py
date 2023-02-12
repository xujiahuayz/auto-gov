import logging
from os import path

import matplotlib.pyplot as plt
import numpy as np

from market_env.constants import FIGURES_PATH
from market_env.env import DefiEnv, PlfPool, PriceDict, User
from rl.dqn_gov import Agent
from rl.rl_env import ProtocolEnv
from run_results.plotting import plot_learning_curve


def training(
    initial_collateral_factor: float = 0.8,
    max_steps: int = 30,
    gamma: float = 0.99,
    n_games: int = 2_000,
    epsilon: float = 1,
    eps_end: float = 0.01,
    eps_dec: float = 5e-5,
    bath_size: int = 128,
    lr: float = 0.003,
    tkn_volatility: float = 2,
    init_saftey_borrow_margin: float = 0.5,
    init_saftey_supply_margin: float = 0.5,
) -> tuple[list[float], list[float], dict[str, list[float]]]:
    # initialize environment
    # defi_env = DefiEnv(prices=PriceDict({"tkn": 1}))
    # Alice = User(name="alice", env=defi_env, funds_available={"tkn": 10_000})
    # plf = PlfPool(
    #     env=defi_env,
    #     initiator=Alice,
    #     initial_starting_funds=8_000,
    #     asset_name="tkn",
    #     collateral_factor=initial_collateral_factor,
    # )

    defi_env = DefiEnv(
        prices=PriceDict({"tkn": 3, "usdc": 0.1, "weth": 1}), max_steps=max_steps
    )
    Alice = User(
        name="alice",
        env=defi_env,
        funds_available={"tkn": 10_000, "usdc": 200_000, "weth": 20_000},
        safety_borrow_margin=init_saftey_borrow_margin,
        safety_supply_margin=init_saftey_supply_margin,
    )
    tkn_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=5_000,
        asset_name="tkn",
        collateral_factor=initial_collateral_factor,
        initial_asset_volatility=tkn_volatility,
    )
    usdc_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=150_000,
        asset_name="usdc",
        collateral_factor=initial_collateral_factor,
        initial_asset_volatility=0.1,
    )
    weth_plf = PlfPool(
        env=defi_env,
        initiator=Alice,
        initial_starting_funds=15_000,
        asset_name="weth",
        collateral_factor=initial_collateral_factor,
        initial_asset_volatility=0,
    )

    env = ProtocolEnv(defi_env)

    # initialize agent
    agent = Agent(
        gamma=gamma,
        epsilon=epsilon,
        batch_size=bath_size,
        n_actions=env.action_space.n,
        eps_end=eps_end,
        input_dims=env.observation_space.shape,
        lr=lr,
        eps_dec=eps_dec,
    )
    # agent = Agent(state_size, action_size)

    scores, eps_history, collateral_factors = [], [], {}

    for plf in defi_env.plf_pools.values():
        collateral_factors[plf.asset_name] = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation.astype(np.float32))
            # this checks done or not
            observation_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-30:])
        if i % 10 == 0:
            logging.info(
                f"{i} score {score:.2f} average score {avg_score:.2f} epsilon {agent.epsilon:.2f} collateral factor {next(iter(defi_env.plf_pools.values())).collateral_factor:.2f}"
            )
        plf = next(iter(defi_env.plf_pools.values()))
        collateral_factors[plf.asset_name].append(plf.collateral_factor)
    return scores, eps_history, collateral_factors


if __name__ == "__main__":
    # show logging level at info
    logging.basicConfig(level=logging.INFO)
    N_GAMES = 50
    training_scores, training_eps_history, training_collateral_factors = training(
        n_games=N_GAMES, eps_dec=0
    )
    x = [i + 1 for i in range(N_GAMES)]
    filename = path.join(FIGURES_PATH, "defi.png")
    plot_learning_curve(x, training_scores, training_eps_history, filename)
    plt.clf()
    for asset, collateral_factors in training_collateral_factors.items():
        plt.plot(x, collateral_factors, label=asset, alpha=0.5)
    # plt.plot(x, training_collateral_factors)
    plt.savefig(path.join(FIGURES_PATH, "collateral_factors.png"))
