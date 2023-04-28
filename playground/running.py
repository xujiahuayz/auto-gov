import logging
from scripts.config import (
    ATTACK_FUNC,
    TKN_PRICES,
    USDC_PRICES,
    BATCH_SIZE,
    EPS_DEC_FACTOR,
    EPSILON_START,
    EPSILON_DECAY,
    EPSILON_END,
)

from rl.training import training_visualizing


logging.basicConfig(level=logging.INFO)

for attack_function in [
    None,
    ATTACK_FUNC,
]:
    for NUM_STEPS in [30 * 12, 30 * 15]:
        for target_on_point in [0.4, 0.5]:
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
