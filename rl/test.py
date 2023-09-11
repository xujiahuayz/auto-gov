import json
import logging
import numpy as np

from rl.utils import init_env
from rl.utils import save_the_nth_model
from rl.utils import load_saved_model

from market_env.utils import generate_price_series
from market_env.constants import FIGURE_PATH
from market_env.constants import DATA_PATH
from rl.config import (
    TKN_PRICES,
    USDC_PRICES,
    ATTACK_FUNC,
    EPS_DEC_FACTOR,
    EPSILON_DECAY,
    EPSILON_END,
    NUM_STEPS,
    TEST_NUM_STEPS,
    TARGET_ON_POINT,
    GAMMA,
    LEARNING_RATE,
    BATCH_SIZE,
)
from rl.training import training
from rl.main_gov import inference_with_trained_model
from rl.rl_env import ProtocolEnv

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
    return np.random.randint(0, t, 3).tolist()

if __name__ == "__main__":
    # test the trained model on a real-world environment
    test_steps = TEST_NUM_STEPS
    prices = {}
    for asset in ["link", "usdc"]:
        # get price data in json from data folder
        with open(DATA_PATH / f"{asset}.json") as f:
            prices[asset] = [
                w["close"]
                for w in json.load(f)["Data"]["Data"][-(test_steps + 2) :]
            ]

    # init the environment
    test_env = init_env(
        initial_collateral_factor=0.65,
        max_steps=test_steps,
        tkn_price_trend_func=lambda x, y: prices["link"],
        usdc_price_trend_func=lambda x, y: prices["usdc"],
    )
    test_protocol_env = ProtocolEnv(test_env)

    for i in range(91):
        trained_model = load_saved_model(i, "trained_model_")
        (
            test_scores,
            test_states,
            test_policies,
            test_rewards,
            test_bench_states,
        )= inference_with_trained_model(
            model=trained_model,
            env=test_protocol_env,
            num_test_episodes=1,
            agent_args={
                "eps_dec": EPSILON_DECAY,
                "eps_end": EPSILON_END,
                "lr": LEARNING_RATE,
                "gamma": GAMMA,
                "epsilon": 1,
                "batch_size": BATCH_SIZE,
                "target_on_point": TARGET_ON_POINT,
            },
        )
        print(f"model {i}: {test_scores}")
    
    # print(f"test_scores: {test_scores}")
    # print(f"test_rewards: {test_rewards}")
    # print(f"test_policies: {test_policies}")
    # print(f"test_states: {test_states}")
    # print(f"test_bench_states: {test_bench_states}")