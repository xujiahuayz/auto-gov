from typing import Callable
import numpy as np
from collections.abc import MutableMapping

from market_env.constants import DEBT_TOKEN_PREFIX, INTEREST_TOKEN_PREFIX


class PriceDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.__dict__ = dict()

        # calls newly written `__setitem__` below
        self.update(*args, **kwargs)

    # The next five methods are requirements of the ABC.
    def __setitem__(self, key: str, value: float):
        if INTEREST_TOKEN_PREFIX in key or DEBT_TOKEN_PREFIX in key:
            raise ValueError("can only set underlying price")
        self.__dict__[key] = value
        self.__dict__[INTEREST_TOKEN_PREFIX + key] = value
        self.__dict__[DEBT_TOKEN_PREFIX + key] = -value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self):
        """echoes class, id, & reproducible representation in the REPL"""
        return f"{self.__dict__}"


def generate_price_series(
    mu: Callable,
    volatility_func: Callable,
    time_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generates a time series of prices following geometric Brownian motion
    with time-variant volatility.

    Args:
    drift (float): The constant drift rate of the asset.
    volatility_func (function): A function that takes in time and returns the
        volatility of the asset at that time.
    time_steps (int): The number of time steps to simulate.
    time_horizon (float): The total time horizon of the simulation.

    Returns:
    numpy.ndarray: A numpy array of shape (time_steps,) containing the simulated
    price series.
    """

    if seed is not None:
        np.random.seed(seed)
    time_array = np.linspace(0, time_steps, time_steps + 1)

    W = np.random.standard_normal(size=time_steps + 1)
    W[0] = 0
    W = np.cumsum(W)
    sigmas = np.array([volatility_func(t) for t in range(time_steps + 1)])
    mus = np.array([mu(t) for t in range(time_steps + 1)])

    drifts = (mus - 0.5 * sigmas**2) * time_array
    diffusion = sigmas * W
    return np.exp(drifts + diffusion)


def borrow_lend_rates(
    util_rate: float,
    spread: float = 0.2,
    rb_factor: float = 20,
) -> tuple[float, float]:
    # TODO: check where to put the factors
    """
    calculate borrow and supply rates based on utilization ratio
    with an arbitrarily-set shape
    """
    # theoretically unnecessary, but to avoid floating point errors
    if util_rate == 0:
        return 0, 0

    assert (
        -1e-9 < util_rate
    ), f"utilization ratio must be non-negative, but got {util_rate}"
    constrained_util_rate = max(0, min(util_rate, 0.97))

    borrow_rate = constrained_util_rate / (rb_factor * (1 - constrained_util_rate))
    daily_borrow_interest = (1 + borrow_rate) ** (1 / 365) - 1
    # this is to make sure that the borrow rate income is able to cover the supply interest expenses
    daily_supply_interest = daily_borrow_interest * constrained_util_rate
    supply_rate = ((1 + daily_supply_interest) ** 365 - 1) * (1 - spread)

    return borrow_rate, supply_rate


if __name__ == "__main__":
    # plot the time series
    import matplotlib.pyplot as plt

    # Generate the price series
    price_series = generate_price_series(
        mu=lambda t: 0.01 * t / 20,
        volatility_func=lambda t: 0.2 * t**1.5 / 1000,
        time_steps=365,
        seed=42,
    )

    # Plot the price series
    plt.plot(price_series)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

    # plot daily log return of the price series
    plt.plot(np.diff(np.log(price_series)))
    plt.xlabel("Time")
    plt.ylabel("Daily Log Return")
    plt.show()

    # test borrow_lend_rates by plotting the rates
    util_rates = np.linspace(0, 1, 100)
    borrow_rates, supply_rates = zip(*[borrow_lend_rates(u) for u in util_rates])
    plt.plot(util_rates, borrow_rates, label="borrow rate")
    plt.plot(util_rates, supply_rates, label="supply rate")
    plt.xlabel("Utilization Ratio")
    plt.ylabel("Interest Rate")
    plt.legend()
    plt.show()
