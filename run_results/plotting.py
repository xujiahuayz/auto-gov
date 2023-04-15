"""
Plotting results from training
"""


import matplotlib.pyplot as plt
import numpy as np


def plot_learning(scores, filename: str, x=int | None, window: int = 5):
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
    plt.savefig(filename)


def plot_time_cdf(times1, times2, times3, bin, filename: str):
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

    size = 13
    fig, ax = plt.subplots(figsize=(6, 3.87))
    ax.plot(bins_count1[1:], cdf1, color="r", linewidth=3, label="30 action/step")
    ax.plot(bins_count2[1:], cdf2, color="g", linewidth=3, label="45 action/step")
    ax.plot(bins_count3[1:], cdf3, color="b", linewidth=3, label="60 action/step")
    ax.set_ylabel("CDF", fontsize=size)
    ax.set_xlabel("Training time per step (ms)", fontsize=size)
    ax.tick_params(axis="both", which="major", labelsize=size)
    # ax.set_xlim(0, 400)
    ax.legend(fontsize=size)
    # ax.grid(True)
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == "__main__":
    pass
