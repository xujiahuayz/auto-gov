import numpy as np

from market_env.utils import generate_price_series

NUM_STEPS = int(30 * 20)
EPSILON_END = 2e-3
EPSILON_DECAY = 5e-7
BATCH_SIZE = 128
EPSILON_START = 1.0
TARGET_ON_POINT = 0.2
EPS_DEC_FACTOR = 0.3
GAMMA = 0.95
LEARNING_RATE = 0.001


def TKN_PRICES(time_steps: int, seed: int | None = None) -> np.ndarray:
    series = generate_price_series(
        time_steps=time_steps,
        seed=seed,
        mu_func=lambda t: 0.00001,
        sigma_func=lambda t: 0.05 + ((t - 200) ** 2) / 5e5,
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
