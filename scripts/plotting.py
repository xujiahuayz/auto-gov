from typing import Any, Callable

# plot time series of collateral factor.
import matplotlib.pyplot as plt
import numpy as np

from market_env.constants import FIGURES_PATH
from market_env.utils import generate_price_series
from rl.main_gov import train_env


def plot_training(
    number_steps: int,
    epsilon_decay: float,
    epsilon_start: float,
    target_on_point: float,
    attack_func: Callable | None,
    eps_history: list[float],
    scores: list[float],
) -> None:
    # plot scores on the left axis and epsilons on the right axis
    score_color = "blue"
    epsilon_color = "orange"
    number_episodes = len(scores)
    attack_on = attack_func is not None
    # TODO: make font size large
    fig, ax1 = plt.subplots()
    # TODO: put specs text inside the plot
    specs_text = f"max steps / episode: {number_steps} \n attacks on: {attack_on}"
    plt.title(specs_text)
    plt.xlim(0, number_episodes - 1)
    ax2 = ax1.twinx()
    ax1.plot(range(number_episodes), scores, color=score_color)
    ax2.plot(range(number_episodes), eps_history, color=epsilon_color)

    # ax2.hlines(
    #     y=[target_on_point],
    #     xmin=[(epsilon_start - target_on_point) / epsilon_decay / number_steps - 0.5],
    #     xmax=[number_episodes],
    #     colors=[epsilon_color],
    #     linestyles="dashed",
    # )

    # TODO: specify when target is turned on by labeling it on ax2, consider logging y

    if attack_on:
        pass

    # label axes
    ax1.set_xlabel("episode")
    ax1.set_ylabel("score", color=score_color)
    ax2.set_ylabel("episode-end $\epsilon$", color=epsilon_color)
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(
        fname=FIGURES_PATH / f"{number_steps}_{target_on_point}_{attack_on}.pdf"
    )
    plt.show()


def plot_examp_states(example_states: list[dict[str, Any]], bs: list[dict[str, Any]]):
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
            range(len(bs)),
            [state["pools"][asset]["reserve"] for state in bs],
            alpha=0.5,
            label=asset,
            color=ASSET_COLORS[asset],
        )
        # plot utilization ratio
        ax3.plot(
            [state["pools"][asset]["utilization_ratio"] for state in bs],
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
    ax.plot(
        [state["net_position"] for state in bs],
        label="benchmark",
        lw=2,
    )

    ax.set_xlabel("time")
    ax.set_ylabel("total net position")
    # legend outside the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

    ax.plot(total_net_position, label="RL")
    ax.set_xlabel("time")
    ax.set_ylabel("total net position")
    # set the legend outside the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)


# color scheme for the three assets
ASSET_COLORS = {
    "tkn": "tab:blue",
    "weth": "tab:orange",
    "usdc": "tab:green",
}


stable_start = int(target_on_point * number_games)

stable_scores = scores[stable_start:]
# find out the position or index of the median score
median_score = sorted(stable_scores, reverse=True)[len(stable_scores) // 2000]
# find out the index of the median score
median_score_index = stable_scores.index(median_score)

example_state = states[stable_start:][median_score_index]

bs = bench_states[stable_start:][median_score_index]
