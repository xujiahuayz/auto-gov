import numpy as np
import pytest
from market_env.env import Plf, User


def test_encode_user_state(
    market_env: Plf, alice: str, bob: str, dai_market_name: str, bat_market_name: str
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


def test_borrow_stable(market_env: Plf, alice: str, dai_market_name: str):
    alice_dai = market_env.markets[dai_market_name].users[alice]

    # 6 months have passed
    market_env.timestamp = SECONDS_PER_YEAR
    alice_dai.stable_timestamp = SECONDS_PER_YEAR // 2

    alice_dai.stable_rate = 0.02
    alice_dai.stable_borrow = 100

    stable_borrow = alice_dai.stable_borrow
    assert stable_borrow == pytest.approx(101.0, rel=0.01)
