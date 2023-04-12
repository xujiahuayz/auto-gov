import logging
from rl.main_gov import training
from run_results.plotting import plot_learning_curve


logging.basicConfig(level=logging.INFO)

scores, eps_history, states, time_cost = training(
    initial_collateral_factor=0.7,
    max_steps=60,
    n_games=1_500,
    lr=0.02,
    eps_end=0.01,
    eps_dec=1e-4,
    batch_size=64,
    tkn_volatility=5,
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
