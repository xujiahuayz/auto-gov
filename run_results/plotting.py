"""
Plotting results from training
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


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


def plot_learning_curve(
    x, scores, epsilons, filename: str, title: str, lines=None
) -> None:
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

    # plt.title(title)
    # # grid
    # plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
