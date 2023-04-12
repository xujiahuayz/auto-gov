import logging
from run_results.running import save_results, plot_results
from market_env.constants import DATA_PATH
import pickle

logging.basicConfig(level=logging.INFO)
save_results(
    initial_collateral_factors=[0.7],
    max_steps_values=[45],
    lr_values=[0],
    batch_size_values=[64],
    n_games_values=[1],
    eps_dec_values=[0],
    eps_end_values=[1],
)
with open(DATA_PATH / "pickle_file_suffix.txt", "r", encoding="utf-8") as f:
    suffix = f.read()
with open(DATA_PATH / f"results-{suffix}.pkl", "rb") as f:
    results = pickle.load(f)
plot_results(results)
