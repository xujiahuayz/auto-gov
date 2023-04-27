import logging
import math
from typing import Callable
from scripts.plotting import plot_training

# plot time series of collateral factor.
import numpy as np

from market_env.utils import generate_price_series
from rl.main_gov import train_env


def training_visualizing(
    number_steps: int,
    epsilon_end: float,
    epsilon_decay: float,
    batch_size: int,
    epsilon_start: float,
    target_on_point: float,
    eps_dec_decrease_with_target: float,
    tkn_prices: Callable,
    usdc_prices: Callable,
    attack_func: Callable | None,
):
    number_episodes = int(
        math.ceil(
            (
                (epsilon_start - target_on_point) / epsilon_decay
                + (target_on_point - epsilon_end)
                / (epsilon_decay * eps_dec_decrease_with_target)
            )
            / number_steps
        )
    )

    logging.info(f"number of episodes: {number_episodes}")

    agent_vars = {
        "gamma": 0.95,
        "epsilon": epsilon_start,
        "lr": 0.00015,
        "eps_end": epsilon_end,
        "eps_dec": epsilon_decay,
        "batch_size": batch_size,
        "target_on_point": target_on_point,
        "eps_dec_decrease_with_target": eps_dec_decrease_with_target,
    }

    (
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_models,
    ) = train_env(
        n_episodes=number_episodes,
        compared_to_benchmark=True,
        agent_args=agent_vars,
        # args for init_env
        max_steps=number_steps,
        initial_collateral_factor=0.75,
        tkn_price_trend_func=tkn_prices,
        usdc_price_trend_func=usdc_prices,
        attack_steps=attack_func,
    )

    plot_training(
        number_steps=number_steps,
        epsilon_decay=epsilon_decay,
        epsilon_start=epsilon_start,
        target_on_point=target_on_point,
        attack_func=attack_func,
        eps_history=eps_history,
        scores=scores,
    )

    return scores, eps_history, states, rewards, time_cost, bench_states, trained_models


def training_parallel(args: tuple, **kargs):
    return training_visualizing(
        number_steps=args[1], target_on_point=args[2], attack_func=args[0], **kargs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    number_steps = int(30 * 18)
    EPSILON_END = 5e-5
    EPSILON_DECAY = 3e-4
    batch_size = 128
    EPSILON_START = 1.0
    target_on_point = 0.4
    eps_dec_decrease_with_target = 0.3

    def tkn_prices(time_steps: int, seed: int | None = None) -> np.ndarray:
        series = generate_price_series(
            time_steps=time_steps,
            seed=seed,
            mu_func=lambda t: 0.00001,
            sigma_func=lambda t: 0.05 + ((t - 200) ** 2) ** 0.01 / 20,
        )
        return series

    def usdc_prices(time_steps: int, seed: int | None = None) -> np.ndarray:
        series = generate_price_series(
            time_steps=time_steps,
            seed=None,
            mu_func=lambda t: 0.0001,
            sigma_func=lambda t: 0.05,
        )
        return series

    def attack_func(t: int) -> list[int]:
        attack_steps = np.random.randint(0, t, 3).tolist()
        attack_steps.sort()
        return attack_steps

    for attack_function in [
        None,
        attack_func,
    ]:
        for number_steps in [30 * 12, 30 * 18]:
            for target_on_point in [0.4, 0.5]:
                (
                    scores,
                    eps_history,
                    states,
                    rewards,
                    time_cost,
                    bench_states,
                    trained_model,
                ) = training_visualizing(
                    number_steps=number_steps,
                    epsilon_end=EPSILON_END,
                    epsilon_decay=EPSILON_DECAY,
                    batch_size=batch_size,
                    epsilon_start=EPSILON_START,
                    target_on_point=target_on_point,
                    eps_dec_decrease_with_target=eps_dec_decrease_with_target,
                    tkn_prices=tkn_prices,
                    usdc_prices=usdc_prices,
                    attack_func=attack_function,
                )
