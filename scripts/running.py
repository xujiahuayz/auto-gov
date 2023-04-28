import logging
from scripts.config import attack_func, tkn_prices, usdc_prices

from scripts.training import training_visualizing


logging.basicConfig(level=logging.INFO)

for attack_function in [
    None,
    attack_func,
]:
    for number_steps in [30 * 18]:
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
                number_steps=number_steps,
                epsilon_end=5e-5,
                epsilon_decay=1e-5,
                batch_size=128,
                epsilon_start=1,
                target_on_point=target_on_point,
                eps_dec_decrease_with_target=0.3,
                tkn_prices=tkn_prices,
                usdc_prices=usdc_prices,
                attack_func=attack_function,
            )
