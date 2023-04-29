from typing import Callable
from matplotlib import pyplot as plt
from market_env.constants import FIGURES_PATH
from rl.training import training_visualizing


def plot_training_results(
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
    ) = training_visualizing(
        number_steps=number_steps,
        target_on_point=target_on_point,
        attack_func=attack_func,
        **kwargs,
    )

    #  start plotting training results
    score_color = "blue"
    epsilon_color = "orange"
    attack_on = attack_func is not None

    # create two subplots that share the x axis
    # the two subplots are created on a grid with 1 column and 2 rows
    plt.rcParams.update({"font.size": 16.5})
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    x_range = range(len(scores))

    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax1.twinx()

    ax1.plot(x_range, eps_history, color=epsilon_color)
    ax1.set_ylabel("episode-end $\epsilon$", color=epsilon_color)

    # add a second x axis to the first subplot on the top
    ax4 = ax3.twiny()
    ax3.set_ylabel("score", color=score_color)
    ax4.plot(x_range, scores, color=score_color)
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
    )
    ax2.scatter(x=RL_bust, y=y_bust * len(RL_bust), label="RL", marker=".", color="r")

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
        fname=str(FIGURES_PATH / f"{number_steps}_{target_on_point}_{attack_on}.pdf")
    )
    plt.show()
