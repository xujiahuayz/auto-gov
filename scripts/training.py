import logging
import math
from typing import Any, Callable
from market_env.constants import FIGURES_PATH

from matplotlib import pyplot as plt


from rl.main_gov import train_env
from scripts.config import attack_func, tkn_prices, usdc_prices

# from scripts.plotting import plot_training


def training_visualizing(
    number_steps: int,
    epsilon_end: float,
    epsilon_decay: float,
    batch_size: int,
    epsilon_start: float,
    target_on_point: float,
    eps_dec_decrease_with_target: float,
    tkn_prices: Callable,
    usdc_prices: Callable,
    attack_func: Callable | None,
    models_chosen: list[int] | None = None,
) -> tuple[
    list[float],
    list[float],
    list[list[dict[str, Any]]],
    list[list[float]],
    list[float],
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
    list[float],
]:
    number_episodes = int(
        math.ceil(
            (
                (epsilon_start - target_on_point) / epsilon_decay
                + (target_on_point - epsilon_end)
                / (epsilon_decay * eps_dec_decrease_with_target)
            )
            / number_steps
        )
    )

    logging.info(f"number of episodes: {number_episodes}")

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
        n_games=number_episodes,
        compared_to_benchmark=True,
        agent_args=agent_vars,
        # args for init_env
        max_steps=number_steps,
        initial_collateral_factor=0.75,
        tkn_price_trend_func=tkn_prices,
        usdc_price_trend_func=usdc_prices,
        attack_steps=attack_func,
    )

    score_color = "blue"
    epsilon_color = "orange"
    attack_on = attack_func is not None

    # create two subplots that share the x axis
    # the two subplots are created on a grid with 1 column and 2 rows
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    x_range = range(number_episodes)
    # make all the fonts bigger
    plt.rcParams.update({"font.size": 16.5})

    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax1.twinx()

    ax1.plot(x_range, eps_history, color=epsilon_color)
    ax1.set_ylabel("episode-end $\epsilon$", color=epsilon_color)

    # add a second x axis to the first subplot on the top
    ax4 = ax3.twiny()
    # make ax3 y lable scientific notation
    # ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax3.set_ylabel("score", color=score_color)
    ax4.plot(x_range, scores, color=score_color)
    ax4.set_xlabel("episode")

    ax2.plot(x_range, losses)
    ax2.set_ylabel("loss")

    bench_bust = [
        x for x in range(len(bench_states)) if len(bench_states[x]) < number_steps
    ]
    RL_bust = [x for x in range(len(states)) if len(states[x]) < number_steps]
    ax2.scatter(
        x=bench_bust,
        y=[1] * len(bench_bust),
        label="benchmark",
        marker="o",
        facecolors="none",
        edgecolors="g",
    )
    ax2.scatter(x=RL_bust, y=[1] * len(RL_bust), label="RL", marker="x", color="r")

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
        fname=FIGURES_PATH / f"{number_steps}_{target_on_point}_{attack_on}.pdf"
    )
    plt.show()

    return (
        scores,
        eps_history,
        states,
        rewards,
        time_cost,
        bench_states,
        trained_models,
        losses,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for attack_function in [
        # None,
        attack_func,
    ]:
        for number_steps in [30 * 18]:
            for target_on_point in [0.4, 0.5]:
                (
                    scores,
                    eps_history,
                    states,
                    rewards,
                    time_cost,
                    bench_states,
                    trained_model,
                    losses,
                ) = training_visualizing(
                    number_steps=number_steps,
                    epsilon_end=5e-5,
                    epsilon_decay=1e-5,
                    batch_size=128,
                    epsilon_start=1,
                    target_on_point=target_on_point,
                    eps_dec_decrease_with_target=0.3,
                    tkn_prices=tkn_prices,
                    usdc_prices=usdc_prices,
                    attack_func=attack_function,
                )
