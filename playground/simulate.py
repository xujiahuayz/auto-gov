import logging

# plot time series of collateral factor.
import matplotlib.pyplot as plt
import numpy as np

from market_env.utils import generate_price_series
from rl.main_gov import train_env


logging.basicConfig(level=logging.INFO)


number_steps = 360 * 2
EPSILON_END = 1e-3
EPSILON_DECAY = 3e-6
batch_size = 64
EPSILON_START = 1.0
number_games = int(
    (EPSILON_START - EPSILON_END) / EPSILON_DECAY / number_steps * 2 // 100 * 100
)


def tkn_prices(time_steps: int, seed: int | None = None) -> np.ndarray:
    series = generate_price_series(
        time_steps=time_steps,
        mu_func=lambda t: 0.01,
        sigma_func=lambda t: 0.01,
        seed=seed,
    )
    # inject sudden price drop
    for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        series[i] = 0.000001
    # inject sudden price rise
    series[20] = 90
    return series


scores, eps_history, states, time_cost, bench_rewards, bench_states = train_env(
    gamma=0.99,
    epsilon=EPSILON_START,
    n_games=number_games,
    lr=0.0015,
    eps_end=EPSILON_END,
    eps_dec=EPSILON_DECAY,
    batch_size=batch_size,
    target_net_enabled=True,
    compared_to_benchmark=True,
    # args for init_env
    max_steps=number_steps,
    initial_collateral_factor=0.7,
    tkn_price_trend_func=tkn_prices,
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
# plot_learning_curve(
#     x=range(len(scores)),
#     scores=scores,
#     epsilons=eps_history,
#     filename=FIGURES_PATH / "test.pdf",
# )

# color scheme for the three assets
ASSET_COLORS = {
    "tkn": "tab:blue",
    "weth": "tab:orange",
    "usdc": "tab:green",
}

stable_start = int(0 * number_games)

stable_scores = scores[stable_start:]
# find out the position or index of the median score
median_score = sorted(stable_scores, reverse=True)[len(stable_scores) // 5000]
# find out the index of the median score
median_score_index = stable_scores.index(median_score)

example_state = states[stable_start:][median_score_index]

# create a figure with two axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for asset in ["tkn", "weth", "usdc"]:
    # plot the collateral factor on the left axis
    ax1.plot(
        [state["pools"][asset]["collateral_factor"] for state in example_state],
        color=ASSET_COLORS[asset],
        label=asset,
    )
    # plot the price on the right axis

    ax2.plot(
        [state["pools"][asset]["price"] for state in example_state],
        color=ASSET_COLORS[asset],
        linestyle="dashed",
    )

# set the labels
ax1.set_ylabel("collateral factor")
ax2.set_ylabel("price")

# set the legend outside the plot
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)


# initialize the figure
fig, ax = plt.subplots()
ax3 = ax.twinx()
for asset in ["tkn", "weth", "usdc"]:
    # plot reserves of each asset in area plot with semi-transparent fill
    ax.fill_between(
        range(len(states[-1])),
        [state["pools"][asset]["reserve"] for state in example_state],
        alpha=0.5,
        label=asset,
        color=ASSET_COLORS[asset],
    )
    # plot utilization ratio
    ax3.plot(
        [state["pools"][asset]["utilization_ratio"] for state in example_state],
        color=ASSET_COLORS[asset],
        linestyle="dotted",
    )
    ax3.set_ylabel("utilization ratio")


# set the labels
ax.set_xlabel("time")
ax.set_ylabel("reserve")
# calculate the env's total net position over time
total_net_position = [state["net_position"] for state in example_state]

# plot the total net position
fig, ax = plt.subplots()

# plot the benchmark case
ax.plot([state["net_position"] for state in bench_states], label="benchmark")
ax.set_xlabel("time")
ax.set_ylabel("total net position")
# legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

ax.plot(total_net_position, label="RL")
ax.set_xlabel("time")
ax.set_ylabel("total net position")
# set the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
