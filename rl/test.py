import json
import logging
import os
import re

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="darkgrid")
sns.set(font_scale=1.4)

from market_env.constants import DATA_PATH, FIGURE_PATH
from market_env.utils import generate_price_series
from rl.config import (
    ATTACK_FUNC,
    BATCH_SIZE,
    EPS_DEC_FACTOR,
    EPSILON_DECAY,
    EPSILON_END,
    GAMMA,
    LEARNING_RATE,
    NUM_STEPS,
    TARGET_ON_POINT,
    TEST_NUM_STEPS,
    TKN_PRICES,
    USDC_PRICES,
)
from rl.main_gov import inference_with_trained_model
from rl.rl_env import ProtocolEnv
from rl.training import training
from rl.utils import init_env, load_saved_model, save_the_nth_model, load_saved_model_fullname


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


def test_model_group():
    # test the trained model on a real-world environment
    test_steps = TEST_NUM_STEPS
    prices = {}
    for asset in ["link", "usdc"]:
        # get price data in json from data folder
        with open(DATA_PATH / f"{asset}.json") as f:
            prices[asset] = [
                w["close"] for w in json.load(f)["Data"]["Data"][-(test_steps + 2) :]
            ]
    
    # Get all filenames in the directory
    all_files = os.listdir(DATA_PATH)
    # get all the file name start with "trained_model_XX_" and end with ".pkl" in the data folder
    model_files = [f for f in all_files if f.startswith('trained_model_256_') and f.endswith('.pkl')]
    
    # pattern = re.compile(r'^trained_model_(\d+)\.pkl$')
    # model_files = [f for f in all_files if pattern.match(f)]
    
    # Output the filenames
    print(model_files)


    # init the environment
    test_env = init_env(
        initial_collateral_factor=0.8,
        max_steps=test_steps,
        tkn_price_trend_func=lambda x, y: prices["link"],
        usdc_price_trend_func=lambda x, y: prices["usdc"],
    )
    test_protocol_env = ProtocolEnv(test_env)

    for i in model_files:
        trained_model = load_saved_model_fullname(DATA_PATH / i)
        (
            test_scores,
            test_states,
            test_policies,
            test_rewards,
            test_bench_states,
            test_bench_2_states,
        ) = inference_with_trained_model(
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


def test_single_model(model_name, initial_cf=0.8):
    test_steps = TEST_NUM_STEPS

    # init the price data
    prices = {}
    for asset in ["link", "usdc"]:
        # get price data in json from data folder
        with open(DATA_PATH / f"{asset}.json") as f:
            prices[asset] = [
                w["close"] for w in json.load(f)["Data"]["Data"][-(test_steps + 2) :]
            ]
    
    # init the environment
    test_env = init_env(
        initial_collateral_factor=initial_cf,
        max_steps=test_steps,
        tkn_price_trend_func=lambda x, y: prices["link"],
        usdc_price_trend_func=lambda x, y: prices["usdc"],
    )
    test_protocol_env = ProtocolEnv(test_env)

    trained_model = load_saved_model_fullname(DATA_PATH / model_name)
    (
        test_scores,
        test_states,
        test_policies,
        test_rewards,
        test_bench_states,
        test_bench_2_states,
        exogenous_vars,
    ) = inference_with_trained_model(
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

    # print(test_bench_states)
    # print(test_bench_2_states)
    # print(test_states)

    example_state = test_states[-1]
    example_exog_vars = exogenous_vars[-1]
    bench_state = test_bench_states[-1]

    # for i in example_exog_vars:
        # print(i)
        # print(example_exog_vars[i])

    # color scheme for the three assets
    ASSET_COLORS = {
        "tkn": ("blue", "/"),
        "usdc": ("green", "\\"),
        "weth": ("orange", "|"),
    }

    # create 2 subfigures that share the x axis
    fig, ax_21 = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax1 = ax_21[0]
    ax2 = ax_21[1]
    for asset, style in ASSET_COLORS.items():
        if asset == "weth":
            log_return = [0] * len(example_exog_vars["tkn_price_trend"])
            # print(log_return)
        else:
            log_return = np.diff(np.log(example_exog_vars[f"{asset}_price_trend"]))
        # print('='*10 + asset + '='*10)
        # print(log_return)
        ax1.plot(
            # calculate log return of the price
            log_return,
            color=style[0],
            label=asset.upper(),
        )
        # plot the collateral factor
        ax2.plot(
            [state["pools"][asset]["collateral_factor"] for state in example_state],
            color=style[0],
            label=asset.upper(),
        )
        ax2.set_ylim(0, 1)

    # set the labels

    x_lable = "step"

    ax1.set_ylabel("Log return of \n price in $\\tt ETH$")
    ax2.set_ylabel("collateral factor")
    ax2.set_xlabel(x_lable)
    # put legend on the top left corner of the plot
    ax1.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(
        fname=str(
            FIGURE_PATH
            / "test_colfact.pdf"
        )
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    test_single_model("trained_model_32_53.pkl", 0.7)