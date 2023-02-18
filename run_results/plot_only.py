import logging
import pickle
import time
from os import path
from itertools import product

from matplotlib import pyplot as plt

from market_env.constants import DATA_PATH, FIGURES_PATH
from rl.main_gov import training
from run_results.plotting import plot_learning_curve
from run_results.running import plot_results


if __name__ == "__main__":
    suffix = "2023-02-13-12-26-44"
    with open(path.join(DATA_PATH, f"results-{suffix}.pkl"), "rb") as f:
        results = pickle.load(f)
    plot_results(results)
