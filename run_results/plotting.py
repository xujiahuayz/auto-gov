"""
Plotting results from training
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_learning(scores, filename: str, x=Optional[int], window: int = 5):
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


def plot_learning_curve(x, scores, epsilons, filename: str | Path, lines=None) -> None:
    fig = plt.figure(figsize=(6, 3.87))
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="black")
    # xlabel and ylabel font size
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="C0")
    # set x and y ticks font size
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 50) : (t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # # tick rotation
    # ax2.tick_params(axis="y", rotation=45)
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.label.set_size(13)
    # set y ticks font size
    ax2.yaxis.set_tick_params(labelsize=13)
    # ticks format scientific notation
    ax2.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


if __name__ == "__main__":
    pass
