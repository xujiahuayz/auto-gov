from collections.abc import MutableMapping

# from market_env.settings import PROJECT_ROOT
import matplotlib.pyplot as plt
from typing import Optional, Literal
from os import path


INTEREST_TOKEN_PREFIX = "interest-"
DEBT_TOKEN_PREFIX = "debt-"


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


def define_price_gov_token(days: int, _start_price: float, _trend_pct: float):
    y = _start_price
    price = [y]

    for _ in range(days):
        y = y * (1 + _trend_pct)
        price.append(y)

    return price
