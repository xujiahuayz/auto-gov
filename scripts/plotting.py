from typing import Callable

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

    ax2.hlines(
        y=[target_on_point],
        xmin=[(epsilon_start - target_on_point) / epsilon_decay / number_steps - 0.5],
        xmax=[number_episodes],
        colors=[epsilon_color],
        linestyles="dashed",
    )

    # TODO: specify when target is turned on by labeling it on ax2, consider logging y

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
