"""
Training defi environment with deep reinforcement learning and plot results
"""

from matplotlib import pyplot as plt
from os import path
import logging
from market_env.constants import FIGURES_PATH
from market_env.env import DefiEnv, PlfPool, PriceDict, User
from rl.dqn_gov import Agent
from rl.main_gov import training
from rl.rl_env import ProtocolEnv
from run_results.plotting import plot_learning_curve


def plot_training_results(
    initial_collateral_factors: list[float] = [0.4, 0.7, 0.99],
    max_steps_values: list[int] = [15, 30, 45],
    lr_values: list[float] = [0.001, 0.003, 0.005],
    n_games_values: list[int] = [1_000, 2_000, 5_000],
):
    # make combinations of different values of initial collateral factor and n_games
    for ms in max_steps_values:
        for icf in initial_collateral_factors:
            for n_game in n_games_values:
                for lr in lr_values:
                    logging.info(
                        f"Training with initial_collateral_factor={icf}, max_steps={ms}, n_games={n_game}, lr={lr}"
                    )
                    scores, eps_history, collateral_factors = training(
                        initial_collateral_factor=icf, max_steps=ms, n_games=n_game
                    )
                    x = [i + 1 for i in range(n_game)]
                    filename = path.join(
                        FIGURES_PATH, f"defi-{icf}-{ms}-{n_game}-{lr}.png"
                    )
                    plot_learning_curve(x, scores, eps_history, filename)
                    plt.clf()
                    plt.plot(x, collateral_factors)
                    plt.savefig(
                        path.join(
                            FIGURES_PATH,
                            f"collateral_factors-{icf}-{ms}-{n_game}-{lr}.png",
                        )
                    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_training_results()
