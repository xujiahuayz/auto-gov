"""
Training defi environment with deep reinforcement learning and plot results
"""

import logging
import pickle
import time
from os import path

from matplotlib import pyplot as plt

from market_env.constants import DATA_PATH, FIGURES_PATH
from rl.main_gov import training
from run_results.plotting import plot_learning_curve


def save_results(
    initial_collateral_factors: list[float] = [0.75],
    max_steps_values: list[int] = [30, 60, 120],
    lr_values: list[float] = [0.001, 0.05, 0.1],
    batch_size_values: list[int] = [64, 128],
    n_games_values: list[int] = [2_000],
    eps_dec_values: list[float] = [1e-5, 5e-5],
    eps_end_values: list[float] = [0.005, 0.01, 0.02],
) -> list[dict]:
    # make combinations of different values of initial collateral factor and n_games
    results = []
    for ms in max_steps_values:
        for icf in initial_collateral_factors:
            for n_game in n_games_values:
                for lr in lr_values:
                    for eps_end in eps_end_values:
                        for eps_dec in eps_dec_values:
                            for batch_size in batch_size_values:
                                logging.info(
                                    f"Training with initial_collateral_factor={icf}, max_steps={ms}, n_games={n_game}, lr={lr}"
                                )
                                (
                                    scores,
                                    eps_history,
                                    training_collateral_factors,
                                ) = training(
                                    initial_collateral_factor=icf,
                                    max_steps=ms,
                                    n_games=n_game,
                                    lr=lr,
                                    eps_end=eps_end,
                                    eps_dec=eps_dec,
                                    batch_size=batch_size,
                                )

                                result = {
                                    "max_steps": ms,
                                    "initial_collateral_factor": icf,
                                    "n_games": n_game,
                                    "lr": lr,
                                    "scores": scores,
                                    "eps_history": eps_history,
                                    "training_collateral_factors": training_collateral_factors,
                                    "es_dec": eps_dec,
                                    "eps_end": eps_end,
                                    "batch_size": batch_size,
                                }
                                results.append(result)

    # pickle results
    pickle_file_suffix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # save this suffix to load the results later
    with open(path.join(DATA_PATH, "pickle_file_suffix.txt"), "w") as f:
        f.write(pickle_file_suffix)
    with open(path.join(DATA_PATH, f"results-{pickle_file_suffix}.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results


def plot_results(results: list[dict]) -> None:
    for result in results:
        ms = result["max_steps"]
        icf = result["initial_collateral_factor"]
        n_game = result["n_games"]
        lr = result["lr"]
        scores = result["scores"]
        eps_history = result["eps_history"]
        batch_size = result["batch_size"]
        eps_dec = result["es_dec"]
        eps_end = result["eps_end"]
        training_collateral_factors = result["training_collateral_factors"]

        filename = path.join(FIGURES_PATH, f"defi-{icf}-{ms}-{n_game}-{lr}.png")
        x = [i + 1 for i in range(n_game)]

        for (
            asset,
            collateral_factors,
        ) in training_collateral_factors.items():
            plt.plot(x, collateral_factors, label=asset, alpha=0.5)
        plt.legend()
        plt.title(
            f"max steps: {ms}, n_games: {n_game}, lr: {lr}, \n eps_end: {eps_end}, eps_dec: {eps_dec}, batch_size: {batch_size}"
        )
        plt.savefig(
            path.join(FIGURES_PATH, f"collateral_factors-{icf}-{ms}-{n_game}-{lr}.png")
        )
        plt.close()

        plot_learning_curve(x, scores, eps_history, filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_results()
    with open(path.join(DATA_PATH, "pickle_file_suffix.txt"), "r") as f:
        suffix = f.read()
    with open(path.join(DATA_PATH, f"results-{suffix}.pkl"), "rb") as f:
        results = pickle.load(f)
    plot_results(results)
