import numpy as np

from market_env.utils import generate_price_series

NUM_STEPS = int(30 * 18)
EPSILON_END = 5e-5
EPSILON_DECAY = 3e-4
BATCH_SIZE = 128
EPSILON_START = 1.0
TARGET_ON_POINT = 0.4
EPS_DEC_FACTOR = 0.3
gamma = 0.95
lr = 0.00015


def TKN_PRICES(time_steps: int, seed: int | None = None) -> np.ndarray:
    series = generate_price_series(
        time_steps=time_steps,
        seed=seed,
        mu_func=lambda t: 0.00001,
        sigma_func=lambda t: 0.05 + ((t - 200) ** 2) ** 0.01 / 20,
    )
    return series


def USDC_PRICES(time_steps: int, seed: int | None = None) -> np.ndarray:
    series = generate_price_series(
        time_steps=time_steps,
        seed=None,
        mu_func=lambda t: 0.0001,
        sigma_func=lambda t: 0.05,
    )
    return series


def ATTACK_FUNC(t: int) -> list[int]:
    attack_steps = np.random.randint(0, t, 3).tolist()
    attack_steps.sort()
    return attack_steps
