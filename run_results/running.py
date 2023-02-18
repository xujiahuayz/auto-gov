"""
Training defi environment with deep reinforcement learning and plot results
"""

import logging
import pickle
import time
from os import path
from itertools import product

from matplotlib import pyplot as plt

from market_env.constants import DATA_PATH, FIGURES_PATH
from rl.main_gov import training
from run_results.plotting import plot_learning_curve


def compute_result(params):
    ms, icf, n_game, lr, eps_end, eps_dec, batch_size = params
    logging.info(
        f"Training with initial_collateral_factor={icf}, max_steps={ms}, n_games={n_game}, lr={lr}, eps_end={eps_end}, eps_dec={eps_dec}, batch_size={batch_size}"
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

    return {
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


def save_results(
    initial_collateral_factors: list[float],
    max_steps_values: list[int],
    lr_values: list[float],
    batch_size_values: list[int],
    n_games_values: list[int],
    eps_dec_values: list[float],
    eps_end_values: list[float],
) -> list[dict]:
    # make combinations of different values of initial collateral factor and n_games
    combinations = list(
        product(
            max_steps_values,
            initial_collateral_factors,
            n_games_values,
            lr_values,
            eps_end_values,
            eps_dec_values,
            batch_size_values,
        )
    )
    print(f"Number of combinations: {len(combinations)}")
    results = list(map(compute_result, combinations))

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

        filename = path.join(
            FIGURES_PATH,
            f"defi-{icf}-{ms}-{n_game}-{lr}-{eps_end}-{eps_dec}-{batch_size}.pdf",
        )
        x = [i + 1 for i in range(n_game)]

        for (
            asset,
            collateral_factors,
        ) in training_collateral_factors.items():
            plt.plot(x, collateral_factors, label=asset, alpha=0.5)
        plt.legend()
        title = f"Collateral factors for {n_game} games, initial collateral factor={icf}, max steps={ms}, lr={lr}, eps_end={eps_end}, eps_dec={eps_dec}, batch_size={batch_size}"
        plt.title(title)
        plt.savefig(
            path.join(
                FIGURES_PATH,
                f"collateral_factors-{icf}-{ms}-{n_game}-{lr}-{eps_end}-{eps_dec}-{batch_size}.pdf",
            )
        )
        plt.show()
        plt.close()

        plot_learning_curve(x, scores, eps_history, filename, title=title)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_results(
        initial_collateral_factors=[0.75],
        max_steps_values=[45],
        lr_values=[0.01],
        batch_size_values=[64],
        n_games_values=[1_000],
        eps_dec_values=[5e-5],
        eps_end_values=[0.03],
    )
    with open(path.join(DATA_PATH, "pickle_file_suffix.txt"), "r") as f:
        suffix = f.read()
    with open(path.join(DATA_PATH, f"results-{suffix}.pkl"), "rb") as f:
        results = pickle.load(f)
    plot_results(results)
