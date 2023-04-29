import json
import logging

# plot time series of collateral factor.
import matplotlib.pyplot as plt
import numpy as np
import math
from market_env.constants import DATA_PATH

from market_env.utils import generate_price_series
from rl.main_gov import inference_with_trained_model, train_env
from rl.rl_env import ProtocolEnv
from rl.utils import init_env


logging.basicConfig(level=logging.INFO)


number_steps = int(30 * 18)
epsilon_end = 1e-4
epsilon_decay = 3e-7
batch_size = 128
epsilon_start = 1.0
target_on_point = 0.4
eps_dec_decrease_with_target = 0.3
number_games = int(
    math.ceil(
        (
            (epsilon_start - target_on_point) / epsilon_decay
            + (target_on_point - epsilon_end)
            / (epsilon_decay * eps_dec_decrease_with_target)
        )
        / number_steps
    )
)

# pick 5 random integers between 0 and number_steps
attack_steps = np.random.randint(0, number_steps, 5).tolist()
# sort the list
attack_steps.sort()

logging.info(f"number of games: {number_games}")

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


(
    scores,
    eps_history,
    states,
    rewards,
    time_cost,
    bench_states,
    trained_models,
    losses,
) = train_env(
    n_episodes=number_games,
    compared_to_benchmark=True,
    agent_args=agent_vars,
    # args for init_env
    max_steps=number_steps,
    initial_collateral_factor=0.75,
    tkn_price_trend_func=tkn_prices,
    usdc_price_trend_func=usdc_prices,
    attack_steps=attack_func,
)


# plot scores on the left axis and epsilons on the right axis
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(range(len(scores)), scores, color="tab:blue")
ax2.plot(range(len(eps_history)), eps_history, color="tab:orange")
# label axes
ax1.set_xlabel("Game")
ax1.set_ylabel("Score", color="tab:blue")
ax2.set_ylabel("Epsilon", color="tab:orange")

plt.show()


# episodes_stored = [w["episode"] for w in trained_models]
# scores_stored = [w["score"] for w in trained_models]
# target_on_episode = int(
#     (epsilon_start - target_on_point) / epsilon_decay / number_steps - 0.5
# )
# # median score before target net
# median_score_before = sorted(stable_scores, reverse=True)[len(stable_scores) // 2000]

# plt.plot(episodes_stored, scores_stored)
# plt.show()
# plt.close()


# test_model = trained_models[-7]


test_steps = 360
prices = {}
for asset in ["link", "usdc"]:
    # get price data in json from data folder
    with open(DATA_PATH / f"{asset}.json", "r") as f:
        prices[asset] = [
            w["close"] for w in json.load(f)["Data"]["Data"][-(test_steps + 2) :]
        ]


test_env = init_env(
    initial_collateral_factor=0.75,
    max_steps=test_steps,
    tkn_price_trend_func=lambda x, y: prices["link"],
    usdc_price_trend_func=lambda x, y: prices["usdc"],
)
test_protocol_env = ProtocolEnv(test_env)

x = inference_with_trained_model(
    model=test_model,
    env=test_protocol_env,
    agent_args=agent_vars,
    num_test_episodes=90,
)
