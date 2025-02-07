import json
import logging
import pickle
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from market_env.constants import DATA_PATH, FIGURE_PATH
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
from rl.utils import init_env, load_saved_model, save_the_nth_model

sns.set_theme(style="darkgrid")
# set font size
plt.rcParams.update({"font.size": 50})


def plot_training_results_seaborn(
    number_steps: int,
    target_on_point: float,
    attack_func: Callable | None,
    constant_col_factor: bool = True,
    **kwargs,
):

    scores = [2, 5, 6, 7, 4, 13, 5]
    eps_history = [0.2, 0.1, 0.01, 0.001, 0, 0, 0]

    losses = range(7)

    # bench_states = range(7)

    # states = range(7)

    # find the index of the last positive score
    last_positive_score: int = next(
        (i for i in reversed(range(len(scores))) if scores[i] > 0), 0
    )

    # select all the scores up to the last positive score
    scores = scores[: last_positive_score + 1]
    # select all the epsilons up to the last positive score
    eps_history = eps_history[: last_positive_score + 1]
    # select all the losses up to the last positive score
    losses = losses[: last_positive_score + 1]

    # NORMALIZE SCORES
    # transform the scores through Hyperbolic tangent function
    scores = np.tanh(scores)
    # TODO: determine the correct normalization method
    # # Linear normalization of scores
    # scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    #  start plotting training results
    score_color = "blue"
    epsilon_color = "orange"
    attack_on = attack_func is not None

    # create two subplots that share the x axis
    # the two subplots are created on a grid with 1 column and 2 rows
    fig, ax = (
        plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 8))
        if attack_on
        else plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 4.5))
    )

    fig.subplots_adjust(hspace=1)

    x_range = range(len(scores))

    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax1.twinx()

    ax1.plot(x_range, eps_history, color=epsilon_color)
    ax1.set_ylabel("episode-end $\epsilon$", color=epsilon_color)

    # add a second x axis to the first subplot on the top
    ax4 = ax3.twiny()
    ax3.set_ylabel(r"$\tanh (\mathrm{score})$", color=score_color)
    ax3.set_ylim(-1.05, 1.05)

    ax3.set_position(ax1.get_position())
    ax4.set_position(ax3.get_position())

    # Add a new parameter for the window size of the rolling mean
    window_size = 50
    # Compute the rolling mean and standard deviation for scores
    scores_series = pd.Series(scores)
    scores_rolling_mean = scores_series.rolling(window=window_size).mean()
    scores_rolling_std = scores_series.rolling(window=window_size).std()

    sns.lineplot(x=x_range, y=scores_rolling_mean, color=score_color, ax=ax4)

    # Add shaded error bounds using the rolling standard deviation
    ax4.fill_between(
        x_range,
        scores_rolling_mean - scores_rolling_std,
        scores_rolling_mean + scores_rolling_std,
        color=score_color,
        alpha=0.1,
        edgecolor="none",
    )
    ax4.set_xlabel("episode")

    ax2.plot(x_range, losses)
    ax2.set_ylabel("loss")
    ax2.set_yscale("log")

    # surpress x-axis numbers but keep the ticks
    plt.setp(ax2.get_xticklabels(), visible=False)

    if attack_on:
        ax_bankrupt = ax[2]

        ax_bankrupt.legend(
            # title="bankruptcy",
            loc="center left",
        )
        # set y axis label

        ax_bankrupt.set_ylabel("bankruptcy density")

        plt.setp(ax_bankrupt.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(
        fname=str(FIGURE_PATH / f"{number_steps}_{target_on_point}_{attack_on}.pdf")
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    batchsize = 32
    print(f"batchsize: {batchsize}")
    for attack_function in [
        # None,
        ATTACK_FUNC,
    ]:
        plot_training_results_seaborn(
            number_steps=NUM_STEPS,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            batch_size=batchsize,
            epsilon_start=1,
            target_on_point=TARGET_ON_POINT,
            eps_dec_decrease_with_target=EPS_DEC_FACTOR,
            tkn_prices=TKN_PRICES,
            usdc_prices=USDC_PRICES,
            attack_func=attack_function,
            PrioritizedReplay_switch=False,
        )
