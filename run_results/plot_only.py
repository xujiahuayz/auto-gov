import pickle
from os import path

from matplotlib import pyplot as plt

from market_env.constants import DATA_PATH, FIGURE_PATH
from run_results.plotting import plot_learning, plot_time_cdf

def plot_training(
    number_steps: int,
    target_on_point: float,
    attack_func: Callable | None,
    **kwargs,
):

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

    # put legend on the bottom of the plot outside of the plot area
    ax2.legend(
        title="bankrupt before episode end",
        bbox_to_anchor=(0, 0),
        loc=2,
        ncol=2,
    )

    # ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(
        fname=str(FIGURE_PATH / f"{number_steps}_{target_on_point}_{attack_on}.pdf")
    )
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
    for result in results:
        scores = result["scores"]
        eps = result["eps_history"]
        loss = result["loss_history"]
    step_num = [i + 1 for i in range(len(scores))]

        

if __name__ == "__main__":
    # draw_delay()
    draw_learning("/Users/yebof/Documents/auto-gov/results_NoAttack_600_0.3_32_3layers.pickle")
