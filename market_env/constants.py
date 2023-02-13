from os import path

from market_env.settings import PROJECT_ROOT

DATA_PATH = path.join(PROJECT_ROOT, "data")
FIGURES_PATH = path.join(PROJECT_ROOT, "figures")

INTEREST_TOKEN_PREFIX = "interest-"
DEBT_TOKEN_PREFIX = "debt-"

PENALTY_REWARD = -200
