"""
Training defi environment with deep reinforcement learning and plot results
"""

import logging
from itertools import product


from market_env.constants import FIGURES_PATH
from rl.main_gov import training
from run_results.plotting import plot_learning_curve


def compute_result(params):
    ms, icf, n_game, lr, eps_end, eps_dec, batch_size = params
    logging.info(
        f"Training with initial_collateral_factor={icf}, max_steps={ms}, n_games={n_game}, lr={lr}, eps_end={eps_end}, eps_dec={eps_dec}, batch_size={batch_size}"
    )
    (scores, eps_history, states, time_cost) = training(
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
        "states": states,
        "es_dec": eps_dec,
        "eps_end": eps_end,
        "batch_size": batch_size,
        "time_cost": time_cost,
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

        filename = (
            FIGURES_PATH
            / f"defi-{icf}-{ms}-{n_game}-{lr}-{eps_end}-{eps_dec}-{batch_size}.pdf"
        )
        x = [i + 1 for i in range(n_game)]

        plot_learning_curve(x, scores, eps_history, filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = save_results(
        initial_collateral_factors=[0.95],
        max_steps_values=[45],
        lr_values=[0.02],
        batch_size_values=[64],
        n_games_values=[200],
        eps_dec_values=[5e-5],
        eps_end_values=[0.02],
    )

    plot_results(results)
