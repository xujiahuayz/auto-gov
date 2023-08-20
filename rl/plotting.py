import logging
from typing import Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from market_env.constants import FIGURE_PATH
from rl.config import (
    ATTACK_FUNC,
    EPS_DEC_FACTOR,
    EPSILON_DECAY,
    EPSILON_END,
    NUM_STEPS,
    TARGET_ON_POINT,
    TKN_PRICES,
    USDC_PRICES,
)
from rl.training import training


sns.set_theme(style="darkgrid")
sns.set(font_scale=1.4)


def plot_training_results_seaborn(
    number_steps: int,
    target_on_point: float,
    attack_func: Callable | None,
    **kwargs,
):
    (
        agent_vars,
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_model,
        losses,
        exogenous_vars,
    ) = training(
        number_steps=number_steps,
        target_on_point=target_on_point,
        attack_func=attack_func,
        **kwargs,
    )

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
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

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

    # Add a new parameter for the window size of the rolling mean
    window_size = 5
    # Compute the rolling mean and standard deviation for scores
    scores_series = pd.Series(scores)
    scores_rolling_mean = scores_series.rolling(window=window_size).mean()
    scores_rolling_std = scores_series.rolling(window=window_size).std()

    # sns.lineplot(x=x_range, y=scores, color=score_color, ax=ax4)
    # ax4.plot(x_range, scores, color=score_color)
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

    y_bust = [min(losses)]
    bench_bust = [
        x for x in range(len(bench_states)) if len(bench_states[x]) < number_steps
    ]
    RL_bust = [x for x in range(len(states)) if len(states[x]) < number_steps]
    ax2.scatter(
        x=bench_bust,
        y=y_bust * len(bench_bust),
        label="benchmark",
        marker="|",
        color="g",
        alpha=0.5,
    )
    ax2.scatter(
        x=RL_bust, y=y_bust * len(RL_bust), label="RL", marker=".", color="r", alpha=0.5
    )

    # surpress x-axis numbers but keep the ticks
    plt.setp(ax2.get_xticklabels(), visible=False)

    # # put legend on the bottom of the plot outside of the plot area
    # ax2.legend(
    #     title="bankrupt before episode end",
    #     bbox_to_anchor=(0, 0),
    #     loc=2,
    #     ncol=2,
    # )

    # if attack fun is not none
    if attack_func is not None:
        # put legend on the top right of the bottom plot inside of the bottom plot area
        ax2.legend(
            title="bankrupt before episode end",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=1,
        )
    else:
        # don't show legend
        ax2.legend().set_visible(False)

    # ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(
        fname=str(FIGURE_PATH / f"{number_steps}_{target_on_point}_{attack_on}.pdf")
    )
    plt.show()
    plt.close()


def plot_example_state(
    number_steps: int,
    target_on_point: float,
    epsilon_start: float,
    epsilon_decay: float,
    bench_score: float = 0,
    **kwargs,
):
    (
        agent_vars,
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_model,
        losses,
        exogenous_vars,
    ) = training(
        number_steps=number_steps,
        target_on_point=target_on_point,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        **kwargs,
    )

    stable_start = int((epsilon_start - target_on_point) / epsilon_decay / number_steps)
    stable_scores = scores[stable_start:]
    # find out the position or index of the top 25 percentile score among all the socres > 0
    example_scores = sorted([x for x in stable_scores if x > bench_score], reverse=True)
    if len(example_scores) == 0:
        raise ValueError("no score above bench_score found")

    example_score = example_scores[len(example_scores) // 4]
    # find out the index of the example score
    good_example_score_index = range(len(states))[stable_start:][
        stable_scores.index(example_score)
    ]

    start_scores = scores[:stable_start]
    bad_example_scores = sorted(start_scores, reverse=True)
    bad_example_score = bad_example_scores[len(bad_example_scores) // 2]
    bad_example_score_index = range(len(states))[:stable_start][
        start_scores.index(bad_example_score)
    ]

    for example_score_index in [good_example_score_index, bad_example_score_index]:
        example_state = states[example_score_index]
        example_exog_vars = exogenous_vars[example_score_index]
        bench_state = bench_states[example_score_index]

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
                log_return = [0] * len(example_exog_vars)
            else:
                log_return = np.diff(np.log(example_exog_vars[f"{asset}_price_trend"]))
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
                / f"colfact{example_score_index}_{number_steps}_{target_on_point}.pdf"
            )
        )
        plt.show()
        plt.close()

        # create 2 subfigures that share the x axis
        fig, ax_2 = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax_20 = ax_2[0]
        ax_21 = ax_2[1]
        # add attack steps from exogenous variables to ax_20 as scatter points
        attack_steps = example_exog_vars["attack_steps"]
        if attack_steps:
            ax_20.scatter(
                x=attack_steps,
                y=[1] * len(attack_steps),
                marker="x",
                color="r",
                label="attack",
            )
            # set the legend for ax_20 above the plot out of the plot area
            ax_20.legend(
                loc="lower left",
            )
        for asset, style in ASSET_COLORS.items():
            # plot utilization ratio
            ax_20.plot(
                [state["pools"][asset]["utilization_ratio"] for state in bench_state],
                color=style[0],
            )
            ax_20.set_ylabel("utilization ratio")
            ax_20.set_ylim(0, 1.1)

            ax_21.fill_between(
                range(len(example_state)),
                [state["pools"][asset]["reserve"] for state in example_state],
                alpha=0.5,
                label=asset.upper(),
                color=style[0],
                # fill pattern
                hatch=style[1],
            )
            # legend on the top left corner of the plot

        ax_21.legend(loc="upper left")

        # set the labels
        ax_21.set_xlabel(x_lable)
        ax_21.set_ylabel("reserve in token units")

        fig.tight_layout()
        fig.savefig(
            fname=str(
                FIGURE_PATH
                / f"state{example_score_index}_{number_steps}_{target_on_point}.pdf"
            )
        )
        plt.show()
        plt.close()

        # calculate the env's total net position over time
        total_net_position = [state["net_position"] for state in example_state]

        # plot the total net position
        fig, ax_np = plt.subplots()

        # plot the benchmark case
        ax_np.plot(
            [state["net_position"] for state in bench_state],
            label="benchmark",
            lw=2,
        )

        ax_np.set_ylabel("total net position in $\\tt ETH$")
        # legend outside the plot
        ax_np.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

        ax_np.plot(total_net_position, label="RL")
        ax_np.set_xlabel(x_lable)
        ax_np.set_ylabel("total net position")
        # set the legend on the top left corner of the plot
        ax_np.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(
            fname=str(
                FIGURE_PATH
                / f"netpos{example_score_index}_{number_steps}_{target_on_point}.pdf"
            )
        )
        plt.show()
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    batchsize = 64
    for attack_function in [
        None,
        # ATTACK_FUNC,
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

        plot_example_state(
            number_steps=NUM_STEPS,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            bench_score=0,
            batch_size=batchsize,
            epsilon_start=1,
            target_on_point=TARGET_ON_POINT,
            eps_dec_decrease_with_target=EPS_DEC_FACTOR,
            tkn_prices=TKN_PRICES,
            usdc_prices=USDC_PRICES,
            attack_func=attack_function,
        )
