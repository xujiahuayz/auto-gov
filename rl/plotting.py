from typing import Callable
from matplotlib import pyplot as plt
from market_env.constants import FIGURES_PATH
from rl.config import ATTACK_FUNC, TARGET_ON_POINT, TKN_PRICES, USDC_PRICES, NUM_STEPS
from rl.training import training


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
    ) = training(
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
        fname=str(FIGURES_PATH / f"{number_steps}_{target_on_point}_{attack_on}.pdf")
    )
    plt.show()


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
    ) = training(
        number_steps=number_steps,
        target_on_point=target_on_point,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        **kwargs,
    )

    # color scheme for the three assets
    ASSET_COLORS = {
        "tkn": "tab:blue",
        "weth": "tab:orange",
        "usdc": "tab:green",
    }

    stable_start = int((epsilon_start - target_on_point) / epsilon_decay / number_steps)
    stable_scores = scores[stable_start:]
    # find out the position or index of the top 25 percentile score among all the socres > 0
    example_scores = sorted([x for x in stable_scores if x > bench_score], reverse=True)
    if len(example_scores) == 0:
        raise ValueError("no score above bench_score found")

    example_score = example_scores[len(example_scores) // 4]
    # find out the index of the median score
    median_score_index = range(len(states))[stable_start:][
        stable_scores.index(example_score)
    ]

    example_state = states[median_score_index]
    bs = bench_states[median_score_index]

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
            [state["pools"][asset]["price"] for state in states[median_score_index]],
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


if __name__ == "__main__":
    for attack_function in [
        None,
        ATTACK_FUNC,
    ]:
        plot_training_results(
            number_steps=NUM_STEPS,
            epsilon_end=5e-5,
            epsilon_decay=1e-4,
            batch_size=128,
            epsilon_start=1,
            target_on_point=TARGET_ON_POINT,
            eps_dec_decrease_with_target=0.3,
            tkn_prices=TKN_PRICES,
            usdc_prices=USDC_PRICES,
            attack_func=attack_function,
        )

        plot_example_state(
            number_steps=NUM_STEPS,
            epsilon_end=5e-5,
            epsilon_decay=1e-4,
            bench_score=-1e5,
            batch_size=128,
            epsilon_start=1,
            target_on_point=TARGET_ON_POINT,
            eps_dec_decrease_with_target=0.3,
            tkn_prices=TKN_PRICES,
            usdc_prices=USDC_PRICES,
            attack_func=attack_function,
        )
