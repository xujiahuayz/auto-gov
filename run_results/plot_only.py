import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from market_env.constants import DATA_PATH, FIGURE_PATH
from run_results.plotting import plot_learning, plot_time_cdf


def plot_training(
    scores: list[float],
    eps_history: list[float],
    losses: list[float],
):
    sns.set_style("darkgrid")
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

    # # log normalization for losses
    # losses = np.log(losses)

    # NORMALIZE SCORES
    # transform the scores through Hyperbolic tangent function
    scores = np.tanh(scores)
    # # Linear normalization of scores
    # scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    #  start plotting training results
    score_color = "blue"
    epsilon_color = "orange"

    # create two subplots that share the x axis
    # the two subplots are created on a grid with 1 column and 2 rows
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    x_range = range(len(scores))

    # ax1 is for the epsilon
    ax1 = ax[0]
    # ax2 is for the loss
    ax2 = ax[1]
    # ax3 is for the score
    ax3 = ax1.twinx()

    ax1.plot(x_range, eps_history, color=epsilon_color)
    ax1.set_ylabel("Epsilon", color=epsilon_color)

    # add a second x axis to the first subplot on the top
    ax4 = ax3.twiny()
    ax3.set_ylabel("Score (normalized)", color=score_color)
    ax3.set_ylim(-1.05, 1.05)

    # Add a new parameter for the window size of the rolling mean
    window_size = 10
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
    ax4.set_xlabel("Episode")

    # ax2.plot(x_range, losses)
    # ax2 log y axis
    ax2.semilogy(x_range, losses)
    ax2.set_ylabel("loss")

    # surpress x-axis numbers but keep the ticks
    plt.setp(ax2.get_xticklabels(), visible=False)

    # put legend on the bottom of the plot outside of the plot area

    # ax2.set_ylim(0, 1)
    fig.tight_layout()
    plt.show()


def plot_results(results: list[dict]) -> None:
    # plot results from training

    for result in results:
        ms = result["max_steps"]
        icf = result["initial_collateral_factor"]
        n_game = result["n_games"]
        lr = result["lr"]
        scores = result["scores"]
        eps_history = result["eps_history"]
        batch_size = result["batch_size"]
        eps_dec = result["es_dec"]
        eps_end = result["eps_end"]
        training_collateral_factors = result["training_collateral_factors"]

        filename = (
            FIGURE_PATH
            / f"defi-{icf}-{ms}-{n_game}-{lr}-{eps_end}-{eps_dec}-{batch_size}.pdf"
        )

        x = [i + 1 for i in range(n_game)]

        # set plt size
        plt.figure(figsize=(6, 3.87))
        for (
            asset,
            collateral_factors,
        ) in training_collateral_factors.items():
            plt.plot(x, collateral_factors, label=asset, alpha=0.5)
        plt.legend()
        # set legend font size
        plt.legend(fontsize=13)
        # set x and y ticks font size
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        # set x label font size
        plt.xlabel("Training steps", fontsize=13)
        # set y label font size
        plt.ylabel("Value of collateral factor", fontsize=13)
        title = f"Collateral factors for {n_game} games, initial collateral factor={icf}, max steps={ms}, lr={lr}, eps_end={eps_end}, eps_dec={eps_dec}, batch_size={batch_size}"
        # plt.title(title)
        plt.tight_layout()
        plt.savefig(
            FIGURE_PATH
            / f"collateral_factors-{icf}-{ms}-{n_game}-{lr}-{eps_end}-{eps_dec}-{batch_size}.pdf"
        )
        plt.show()
        plt.close()

        plot_learning_curve(x, scores, eps_history, filename)


def draw_delay():
    suffix1 = "2023-02-18-00-20-02"
    suffix2 = "2023-02-18-00-21-19"
    suffix3 = "2023-02-18-00-23-03"
    with open(path.join(DATA_PATH, f"results-{suffix1}.pkl"), "rb") as f:
        results1 = pickle.load(f)
    with open(path.join(DATA_PATH, f"results-{suffix2}.pkl"), "rb") as f:
        results2 = pickle.load(f)
    with open(path.join(DATA_PATH, f"results-{suffix3}.pkl"), "rb") as f:
        results3 = pickle.load(f)
    for result in results1:
        times1 = result["time_cost"]
    for result in results2:
        times2 = result["time_cost"]
    for result in results3:
        times3 = result["time_cost"]
    plot_time_cdf(times1, times2, times3, 200, "time_cdf.pdf")


def draw_learning(filename: str):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    print("Input loaded!")

    # 0: agent_vars,
    # 1: scores,
    # 2: eps_history,
    # 3: states,
    # 4: rewards,
    # 5: time_cost,
    # 6: bench_states,
    # 7: trained_models,
    # 8: losses,
    # 9: exogenous_vars,

    scores = results[1]
    print(scores)
    eps = results[2]
    loss = results[8]
    step_num = [i + 1 for i in range(len(scores))]
    plot_training(scores, eps, loss)


if __name__ == "__main__":
    # draw_delay()

    data_path = DATA_PATH / "results-2023-02-18-00-20-02.pkl"
    draw_learning(data_path)
