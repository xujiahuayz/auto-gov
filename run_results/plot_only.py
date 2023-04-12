import pickle
from os import path

from matplotlib import pyplot as plt

from market_env.constants import DATA_PATH, FIGURES_PATH
from run_results.plotting import plot_learning_curve, plot_time_cdf


def plot_results(results: list[dict]) -> None:
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
            FIGURES_PATH
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
            FIGURES_PATH
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


if __name__ == "__main__":
    draw_delay()
