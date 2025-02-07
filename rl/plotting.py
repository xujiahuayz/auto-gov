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
    (
        agent_vars,
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        bench_2_states,
        trained_model,
        losses,
        exogenous_vars,
    ) = training(
        number_steps=number_steps,
        target_on_point=target_on_point,
        attack_func=attack_func,
        constant_col_factor=constant_col_factor,
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
    fig, ax = (
        plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 8))
        if attack_on
        else plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 4.5))
    )

    fig.subplots_adjust(hspace=0.01)

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

        # plot attack density
        bench_bust = [
            x for x in range(len(bench_states)) if len(bench_states[x]) < number_steps
        ]
        RL_bust = [x for x in range(len(states)) if len(states[x]) < number_steps]
        sns.kdeplot(
            bench_bust,
            label="baseline",
            color="#1f77b4",
            alpha=0.5,
            ax=ax_bankrupt,
            clip=(0, len(bench_states)),
        )
        sns.kdeplot(
            RL_bust,
            label="RL",
            color="#ff7f0e",
            alpha=0.5,
            ax=ax_bankrupt,
            clip=(0, len(states)),
        )

        ax_bankrupt.scatter(
            x=bench_bust,
            y=[0.0002] * len(bench_bust),
            marker=3,
            color="#1f77b4",
            alpha=0.5,
        )
        ax_bankrupt.scatter(
            x=RL_bust,
            y=[0.00004] * len(RL_bust),
            marker=2,
            color="#ff7f0e",
            alpha=0.5,
        )

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

    return trained_model


def plot_example_state(
    number_steps: int,
    target_on_point: float,
    epsilon_start: float,
    epsilon_decay: float,
    bench_score: float = 0,
    constant_col_factor: bool = True,
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
        bench_2_states,
        trained_model,
        losses,
        exogenous_vars,
    ) = training(
        number_steps=number_steps,
        target_on_point=target_on_point,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        constant_col_factor=constant_col_factor,
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

    percentile = 8
    print(f"percentile: {percentile}")

    start_scores = scores[:stable_start]
    bad_example_scores = sorted(start_scores, reverse=True)
    bad_example_score = bad_example_scores[len(bad_example_scores) // percentile]
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

        """""
        Price trajectories and collateral factor adjustments of all tokens 
        """ ""
        # create 2 subfigures that share the x axis
        fig, ax_21 = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax1 = ax_21[0]
        ax2 = ax_21[1]
        for asset, style in ASSET_COLORS.items():
            if asset == "weth":
                log_return = [0] * len(example_exog_vars["tkn_price_trend"])
            else:
                log_return = np.diff(np.log(example_exog_vars[f"{asset}_price_trend"]))
            ax1.plot(
                # calculate log return of the price
                log_return,
                color=style[0],
                # if asset == "weth", label="ETH"
                label=asset.upper() if asset != "weth" else "ETH",
            )
            # plot the collateral factor
            ax1.legend(loc="lower center", ncol=3, fontsize=16)
            ax2.plot(
                [state["pools"][asset]["collateral_factor"] for state in example_state],
                color=style[0],
                # if asset == "weth", label="ETH"
                label=asset.upper() if asset != "weth" else "ETH",
            )
            ax2.set_ylim(0, 1)

        # set the labels

        x_lable = "step ($t$)"

        # ax1.set_ylabel("$\ln\frac{P_{t}}{P_{t-1}}$")
        ax1.set_ylabel("$\ln\\frac{P_{t}}{P_{t-1}}$")
        ax2.set_ylabel("$C$")
        ax2.set_xlabel(x_lable)

        ax1.legend(loc="lower center", ncol=3, fontsize=16)
        fig.tight_layout()
        fig.savefig(
            fname=str(
                FIGURE_PATH
                / f"colfact{example_score_index}_{number_steps}_{target_on_point}.pdf"
            )
        )
        plt.show()
        plt.close()

        """""
        Lending pool state over time
        """ ""
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
                loc="lower right",
            )
        for asset, style in ASSET_COLORS.items():
            # plot utilization ratio
            ax_20.plot(
                ###### [state["pools"][asset]["utilization_ratio"] for state in bench_state],
                [state["pools"][asset]["utilization_ratio"] for state in example_state],
                color=style[0],
            )
            ax_20.set_ylabel("$U$")
            ax_20.set_ylim(0, 1.1)

            ax_21.fill_between(
                range(len(example_state)),
                [state["pools"][asset]["reserve"] for state in example_state],
                alpha=0.5,
                label=asset.upper() if asset != "weth" else "ETH",
                color=style[0],
                # fill pattern
                hatch=style[1],
            )

        ax_21.legend(loc="upper left", fontsize=16)
        ax_21.set_ylim(0, 3001)

        # set the labels
        ax_21.set_xlabel(x_lable)
        ax_21.set_ylabel("$W$")

        fig.tight_layout()
        fig.savefig(
            fname=str(
                FIGURE_PATH
                / f"state{example_score_index}_{number_steps}_{target_on_point}.pdf"
            )
        )
        plt.show()
        plt.close()

        """""
        Protocol's total net position 
        """ ""

        # calculate the env's total net position over time
        total_net_position = [state["net_position"] for state in example_state]

        # plot the total net position
        fig, ax_np = plt.subplots()

        # plot the benchmark case
        ax_np.plot(
            [state["net_position"] for state in bench_state],
            label="baseline",
            lw=2,
        )

        # ax_np.set_ylabel("$N$")
        # # legend outside the plot
        # ax_np.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

        ax_np.plot(total_net_position, label="RL")
        ax_np.set_xlabel(x_lable)
        ax_np.set_ylabel("$N$")
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


def save_time():
    logging.basicConfig(level=logging.INFO)
    for step_number in [300, 450, 600, 750]:
        batchsize = 120
        print(f"batchsize: {batchsize}")
        attack_func = None
        target_on_point = 1
        constant_col_factor = 0.75

        (
            agent_vars,
            scores,
            eps_history,
            states,
            rewards,
            time_cost,
            bench_states,
            bench_2_states,
            trained_model,
            losses,
            exogenous_vars,
        ) = training(
            number_steps=step_number,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            epsilon_start=1,
            batch_size=batchsize,
            target_on_point=target_on_point,
            eps_dec_decrease_with_target=EPS_DEC_FACTOR,
            attack_func=attack_func,
            constant_col_factor=constant_col_factor,
            tkn_prices=TKN_PRICES,
            usdc_prices=USDC_PRICES,
            PrioritizedReplay_switch=False,
        )

        # last 300 items
        # time_cost = time_cost[-300:]
        # save time_to_save to a file, one line for one number
        with open(f"time_cost_{step_number}_{target_on_point}.txt", "w") as f:
            f.write(str(time_cost))


# if __name__ == "__main__":
#     save_time()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    batchsize = 32
    print(f"batchsize: {batchsize}")
    for attack_function in [
        # None,
        ATTACK_FUNC,
    ]:
        training_models = plot_training_results_seaborn(
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

        # # chosse a well-trained model and a bad-trained model to plot example state
        # plot_example_state(
        #     number_steps=NUM_STEPS,
        #     epsilon_end=EPSILON_END,
        #     epsilon_decay=EPSILON_DECAY,
        #     bench_score=0,
        #     batch_size=batchsize,
        #     epsilon_start=1,
        #     target_on_point=TARGET_ON_POINT,
        #     eps_dec_decrease_with_target=EPS_DEC_FACTOR,
        #     tkn_prices=TKN_PRICES,
        #     usdc_prices=USDC_PRICES,
        #     attack_func=attack_function,
        # )

        # # save a trained model
        # save_the_nth_model(1, "trained_model_", training_models)
        # save_the_nth_model(3, "trained_model_", training_models)
        # save_the_nth_model(4, "trained_model_", training_models)
        # save_the_nth_model(5, "trained_model_", training_models)
        # # # # #
        # save_the_nth_model(-5, "trained_model_", training_models)
        # save_the_nth_model(-3, "trained_model_", training_models)
        # save_the_nth_model(-1, "trained_model_", training_models)

        # # save all the trained models
        # for i in range(len(training_models)):
        #     save_the_nth_model(i, "trained_model_" + str(batchsize) + "_", training_models)

        # load a trained model
        # trained_model = load_saved_model(31, "trained_model_")
        # print(trained_model)
