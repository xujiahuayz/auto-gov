import logging
from rl.main_gov import init_env, train_env
from run_results.plotting import plot_learning_curve


logging.basicConfig(level=logging.INFO)

sim_env = init_env(
    max_steps=60,
    initial_collateral_factor=0.7,
    tkn_volatility=5,
)


scores, eps_history, states, time_cost = train_env(
    defi_env=sim_env,
    gamma=0.99,
    epsilon=1.0,
    n_games=1_000,
    lr=0.001,
    eps_end=0.01,
    eps_dec=0.999,
    batch_size=64,
)


plot_learning_curve(
    x=range(len(scores)), scores=scores, epsilons=eps_history, filename="test.png"
)
# plot time series of collateral factor
import matplotlib.pyplot as plt

# color scheme for the three assets
ASSET_COLORS = {
    "tkn": "tab:blue",
    "weth": "tab:orange",
    "usdc": "tab:green",
}

# create a figure with two axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for asset in ["tkn", "weth", "usdc"]:
    # plot the collateral factor on the left axis
    ax1.plot(
        [state["pools"][asset]["collateral_factor"] for state in states[-1]],
        color=ASSET_COLORS[asset],
        label=asset,
    )
    # plot the price on the right axis

    ax2.plot(
        [state["pools"][asset]["price"] for state in states[-1]],
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
        range(len(states[-1])),
        [state["pools"][asset]["reserve"] for state in states[-1]],
        alpha=0.5,
        label=asset,
        color=ASSET_COLORS[asset],
    )
    # plot total_i_token in line plot
    ax3.plot(
        [state["pools"][asset]["borrow_apy"] for state in states[-1]],
        color=ASSET_COLORS[asset],
        linestyle="dashed",
    )
    ax3.plot(
        [state["pools"][asset]["supply_apy"] for state in states[-1]],
        color=ASSET_COLORS[asset],
        linestyle="dotted",
    )


# set the labels
ax.set_xlabel("time")
ax.set_ylabel("reserve")

# set the legend outside the plot
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
