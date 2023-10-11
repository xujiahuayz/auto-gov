"""
Plotting results from training
"""


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_learning(scores, filename: str, x=int | None, window: int = 5):
    # plot the scores
    # for policy gradient algorithms

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window) : (t + 1)])
    if x is None:
        x = [i + 1 for i in range(N)]
    plt.ylabel("Score")
    plt.xlabel("Game")
    plt.plot(x, running_avg)
    plt.show()
    plt.savefig(filename)


def plot_time_cdf(times1, times2, times3, times4, bin, filename: str):
    # plot cdf of time cost

    sns.set()
    curve1 = [t * 1000 for t in times1]
    curve1 = np.asarray(curve1)
    count, bins_count1 = np.histogram(curve1, bins=bin)
    curve1 = count / sum(count)
    cdf1 = np.cumsum(curve1)

    curve2 = [t * 1000 for t in times2]
    curve2 = np.asarray(curve2)
    count, bins_count2 = np.histogram(curve2, bins=bin)
    curve2 = count / sum(count)
    cdf2 = np.cumsum(curve2)

    curve3 = [t * 1000 for t in times3]
    curve3 = np.asarray(curve3)
    count, bins_count3 = np.histogram(curve3, bins=bin)
    curve3 = count / sum(count)
    cdf3 = np.cumsum(curve3)

    curve4 = [t * 1000 for t in times4]
    curve4 = np.asarray(curve4)
    count, bins_count4 = np.histogram(curve4, bins=bin)
    curve4 = count / sum(count)
    cdf4 = np.cumsum(curve4)

    size = 16
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(bins_count1[1:], cdf1, color="r", linewidth=3, label="300 action/eps")
    ax.plot(bins_count2[1:], cdf2, color="g", linewidth=3, label="450 action/eps")
    ax.plot(bins_count3[1:], cdf3, color="b", linewidth=3, label="600 action/eps")
    ax.plot(bins_count4[1:], cdf4, color="#8c564b", linewidth=3, label="750 action/eps")
    ax.set_ylabel("CDF", fontsize=size)
    ax.set_xlabel("Training time per episode (ms)", fontsize=size)
    ax.tick_params(axis="both", which="major", labelsize=size)
    # ax.set_xlim(0, 400)
    ax.legend(loc="lower right", fontsize=size-2)
    # ax.grid(True)
    fig.tight_layout()
    # plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == "__main__":
    with open("time_cost_300_1.txt", "r") as f:
        time_cost_300_no_target = eval(f.read())
    with open("time_cost_450_1.txt", "r") as f:
        time_cost_450_no_target = eval(f.read())
    with open("time_cost_600_1.txt", "r") as f:
        time_cost_600_no_target = eval(f.read())
    with open("time_cost_750_1.txt", "r") as f:
        time_cost_750_no_target = eval(f.read())

    plot_time_cdf(
        time_cost_300_no_target,
        time_cost_450_no_target,
        time_cost_600_no_target,
        time_cost_750_no_target,
        600,
        "time_cost_no_target.png",
    )

    # with open("time_cost_300_target.txt", "r") as f:
    #     time_cost_300_target = eval(f.read())
    # with open("time_cost_450_target.txt", "r") as f:
    #     time_cost_450_target = eval(f.read())
    # with open("time_cost_600_target.txt", "r") as f:
    #     time_cost_600_target = eval(f.read())

    # plot_time_cdf(time_cost_300_target, time_cost_450_target, time_cost_600_target, 500, "time_cost_target.png")
