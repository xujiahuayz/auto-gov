import logging
import math
from typing import Any, Callable

from rl.config import ATTACK_FUNC, GAMMA, LEARNING_RATE, TKN_PRICES, USDC_PRICES
from rl.main_gov import train_env


def training(
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
    constant_col_factor: bool = True,
    PrioritizedReplay_switch: bool = False,
):
    number_episodes = int(
        math.ceil(
            (
                (epsilon_start - target_on_point) / epsilon_decay
                + (target_on_point - epsilon_end)
                / (epsilon_decay * eps_dec_decrease_with_target)
            )
            / number_steps
            * 1.5
        )
    )
    # number_episodes = 150

    logging.info(f"number of episodes: {number_episodes}")

    agent_vars = {
        "gamma": GAMMA,
        "epsilon": epsilon_start,
        "lr": LEARNING_RATE,
        "eps_end": epsilon_end,
        "eps_dec": epsilon_decay,
        "batch_size": batch_size,
        "target_on_point": target_on_point,
        "eps_dec_decrease_with_target": eps_dec_decrease_with_target,
        "PrioritizedReplay_switch": PrioritizedReplay_switch,
    }

    (
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        bench_2_states,
        trained_models,
        losses,
        exogenous_vars,
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
        constant_col_factor=constant_col_factor,
    )

    return (
        agent_vars,
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        bench_2_states,
        trained_models,
        losses,
        exogenous_vars,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for attack_function in [
        # None,
        ATTACK_FUNC,
    ]:
        for NUM_STEPS in [30 * 10, 30 * 15, 30 * 20]:
            for target_on_point in [0.2]:
                results_unpacked = training(
                    number_steps=NUM_STEPS,
                    epsilon_end=5e-5,
                    epsilon_decay=1e-6,
                    batch_size=128,
                    epsilon_start=1,
                    target_on_point=target_on_point,
                    eps_dec_decrease_with_target=0.3,
                    tkn_prices=TKN_PRICES,
                    usdc_prices=USDC_PRICES,
                    attack_func=attack_function,
                )

                # time_cost = results_unpacked[5]
                # with open(f"time_cost_{NUM_STEPS}.txt", "w") as f:
                #     f.write(str(time_cost))
