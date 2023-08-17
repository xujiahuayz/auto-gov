import logging
import multiprocessing
import pickle
from typing import Callable

from rl.config import (
    ATTACK_FUNC,
    EPS_DEC_FACTOR,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    TKN_PRICES,
    USDC_PRICES,
)
from rl.training import training

logging.basicConfig(level=logging.INFO)


def run_training_visualizing(params: tuple[Callable | None, int, float, int]):
    attack_function, num_steps, target_on_point, batch_size = params
    return training(
        number_steps=num_steps,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=batch_size,
        epsilon_start=EPSILON_START,
        target_on_point=target_on_point,
        eps_dec_decrease_with_target=EPS_DEC_FACTOR,
        tkn_prices=TKN_PRICES,
        usdc_prices=USDC_PRICES,
        attack_func=attack_function,
    )


if __name__ == "__main__":
    param_combinations = [
        (this_attack_function, this_num_steps, this_target_on_point, this_batch_size)
        # for attack_function in [None, ATTACK_FUNC]
        # for this_attack_function in [ATTACK_FUNC]
        for this_attack_function in [None]
        # for NUM_STEPS in [30 * 12, 30 * 15]
        for this_num_steps in [30 * 20]
        # for target_on_point in [0.4, 0.5]
        for this_target_on_point in [0.3]
        for this_batch_size in [32, 64, 128, 256, 512]
    ]

    print(f"Running {len(param_combinations)} combinations of parameters")

    # with multiprocessing.Pool() as pool:
    #     pool.map(run_training_visualizing, param_combinations)
    # do not use multiprocessing, it will cause the program to crash.
    # let it run one by one.

    for params in param_combinations:
        results = run_training_visualizing(params)

        # store results to file
        attack_function, num_steps, target_on_point, batch_size = params
        attack_str = "WithAttack" if attack_function else "NoAttack"
        filename = (
            f"results_{attack_str}_{num_steps}_{target_on_point}_{batch_size}.pickle"
        )

        with open(filename, "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {filename}")
