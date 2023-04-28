import logging
import multiprocessing

from scripts.config import (
    ATTACK_FUNC,
    BATCH_SIZE,
    EPS_DEC_FACTOR,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    TKN_PRICES,
    USDC_PRICES,
)
from rl.training import training_visualizing

logging.basicConfig(level=logging.INFO)


def run_training_visualizing(params):
    attack_function, NUM_STEPS, target_on_point = params
    (
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_model,
        losses,
    ) = training_visualizing(
        number_steps=NUM_STEPS,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        target_on_point=target_on_point,
        eps_dec_decrease_with_target=EPS_DEC_FACTOR,
        tkn_prices=TKN_PRICES,
        usdc_prices=USDC_PRICES,
        attack_func=attack_function,
    )
    return (
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_model,
        losses,
    )


if __name__ == "__main__":
    param_combinations = [
        (attack_function, NUM_STEPS, target_on_point)
        for attack_function in [None, ATTACK_FUNC]
        for NUM_STEPS in [30 * 12, 30 * 15]
        for target_on_point in [0.4, 0.5]
    ]

    with multiprocessing.Pool() as pool:
        pool.map(run_training_visualizing, param_combinations)

    # `results` now contains the results for all parameter combinations
