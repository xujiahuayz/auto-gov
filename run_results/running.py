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


def plot_training_results(n_games_values: list[int] = [1_000, 2_000, 5_000]):
    for n_game in n_games_values:
        scores, eps_history, collateral_factors = training(n_games=n_game)
        x = [i + 1 for i in range(n_game)]
        filename = path.join(FIGURES_PATH, f"defi-{n_game}.png")
        plot_learning_curve(x, scores, eps_history, filename)
        plt.clf()
        plt.plot(x, collateral_factors)
        plt.savefig(path.join(FIGURES_PATH, f"collateral_factors-{n_game}.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_training_results()
