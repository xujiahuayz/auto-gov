"""
Training defi environment with deep reinforcement learning and plot results
"""

import pickle
from matplotlib import pyplot as plt
from os import path
import logging
from market_env.constants import DATA_PATH, FIGURES_PATH
from rl.main_gov import training
from run_results.plotting import plot_learning_curve


def save_results(
    initial_collateral_factors: list[float] = [0.75],
    max_steps_values: list[int] = [120],
    lr_values: list[float] = [0.06],
    n_games_values: list[int] = [3_000],
    eps_dec_values: list[float] = [5e-5],
    eps_end_values: list[float] = [0.001],
) -> list[dict]:
    # make combinations of different values of initial collateral factor and n_games
    results = []
    for ms in max_steps_values:
        for icf in initial_collateral_factors:
            for n_game in n_games_values:
                for lr in lr_values:
                    logging.info(
                        f"Training with initial_collateral_factor={icf}, max_steps={ms}, n_games={n_game}, lr={lr}"
                    )
                    scores, eps_history, training_collateral_factors = training(
                        initial_collateral_factor=icf,
                        max_steps=ms,
                        n_games=n_game,
                        lr=lr,
                        eps_end=eps_dec_values[0],
                        eps_dec=eps_end_values[0],
                    )

                    result = {
                        "max_steps": ms,
                        "initial_collateral_factor": icf,
                        "n_games": n_game,
                        "lr": lr,
                        "scores": scores,
                        "eps_history": eps_history,
                        "training_collateral_factors": training_collateral_factors,
                    }
                    results.append(result)

    # pickle results
    with open(path.join(DATA_PATH, "results.pkl"), "wb") as f:
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
        training_collateral_factors = result["training_collateral_factors"]

        filename = path.join(FIGURES_PATH, f"defi-{icf}-{ms}-{n_game}-{lr}.png")
        x = [i + 1 for i in range(n_game)]
        plot_learning_curve(x, scores, eps_history, filename)
        plt.clf()
        for (
            asset,
            collateral_factors,
        ) in training_collateral_factors.items():
            plt.plot(x, collateral_factors, label=asset, alpha=0.5)
        plt.legend()
        plt.savefig(path.join(FIGURES_PATH, "collateral_factors.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_results()
    with open(path.join(DATA_PATH, "results.pkl"), "rb") as f:
        results = pickle.load(f)
    plot_results(results)
