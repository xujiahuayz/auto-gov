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


def simulate_gbm(S0, mu, sigma, T, N, seed=None):
    """Simulate a time series of prices that follow geometric Brownian motion.

    Parameters
    ----------
    S0 : float
        The initial price of the asset.
    mu : float
        The drift coefficient of the asset.
    sigma : float
        The volatility coefficient of the asset.
    T : float
        The time horizon of the simulation, in years.
    N : int
        The number of time steps in the simulation.
    seed : int, optional
        Seed for the random number generator. Default is None.

    Returns
    -------
    ndarray
        A 1-dimensional array of length N+1 that contains the simulated prices.

    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)
    W = np.random.standard_normal(size=N + 1)
    W[0] = 0
    W = np.cumsum(W) * np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    S = S0 * np.exp(drift + diffusion)
    return S


if __name__ == "__main__":
    # plot the time series
    import matplotlib.pyplot as plt

    plt.plot(simulate_gbm(S0=1, mu=0.05, sigma=0.2, T=1, N=1000, seed=42))
    plt.plot(simulate_gbm(S0=1, mu=0.05, sigma=0, T=1, N=1000, seed=42))
    plt.plot(simulate_gbm(S0=1, mu=0, sigma=0.2, T=1, N=1000, seed=42))
    plt.plot(simulate_gbm(S0=1, mu=0, sigma=1, T=1, N=1000, seed=42))
    plt.title("Geometric Brownian Motion Simulation")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.show()
