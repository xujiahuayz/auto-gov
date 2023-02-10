import numpy as np
import pytest
from market_env.env import PlfPool, User


def test_encode_user_state(
    market_env: PlfPool, alice: str, dai_market_name: str, bat_market_name: str
):
    market_env.markets[bat_market_name].get_user(bob).unsupplied_amount = 0

    alice_state = market_env.encode_user_state(alice)
    bob_state = market_env.encode_user_state(bob)
    assert isinstance(alice_state, np.ndarray)
    assert isinstance(bob_state, np.ndarray)

    # NOTE: index 0 is the unsupplied amount of DAI, which is the only variable
    # that should be different, see conftest.py for how these are initialized
    assert np.array_equal(alice_state[1:], bob_state[1:])

    deposit_action = Deposit(alice, alice, dai_market_name, 10)
    deposit_action.execute(market_env)

    alice_state = market_env.encode_user_state(alice)
    assert not np.array_equal(alice_state[1:], bob_state[1:])
