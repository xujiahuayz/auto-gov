from typing import Optional
import pytest
from plf_env.actions import (
    Borrow,
    Deposit,
    Liquidate,
    Redeem,
    Repay,
    Swap,
    Exchange,
    Transfer,
)
from plf_env.env import PLFEnv


def execute_deposit(
    plf_env: PLFEnv,
    user: str,
    market: str,
    amount: float,
    on_behalf_of: Optional[str] = None,
):
    if on_behalf_of is None:
        on_behalf_of = user
    deposit_action = Deposit(user, on_behalf_of, market, amount)
    deposit_action.execute(plf_env)


def execute_borrow(
    plf_env: PLFEnv,
    user: str,
    market: str,
    amount: float,
    is_stable_borrow: bool = True,
    on_behalf_of: Optional[str] = None,
):
    if on_behalf_of is None:
        on_behalf_of = user
    borrow_action = Borrow(
        user, on_behalf_of, market, amount, is_stable_borrow=is_stable_borrow
    )
    borrow_action.execute(plf_env)


def test_swap_tokens(
    plf_env: PLFEnv, alice: str, dai_market_name: str, bat_market_name
):
    swap_amount = 30

    alice_init_dai_balance = (
        plf_env.markets[dai_market_name].get_user(alice).unsupplied_amount
    )
    swap_action = Exchange(alice, dai_market_name, bat_market_name, swap_amount)
    swap_action.execute(plf_env)

    alice_dai_balance = (
        plf_env.markets[dai_market_name].get_user(alice).unsupplied_amount
    )
    alice_bat_balance = (
        plf_env.markets[bat_market_name].get_user(alice).unsupplied_amount
    )

    assert alice_dai_balance == pytest.approx(alice_init_dai_balance - swap_amount)

    prices = plf_env.external_data.prices
    expected_bat_balance = (
        swap_amount
        * prices[dai_market_name]
        / prices[bat_market_name]
        * (1 - Exchange.SWAP_FEE)
    )
    assert alice_bat_balance == pytest.approx(expected_bat_balance)


def test_deposit(plf_env: PLFEnv, alice: str, dai_market_name: str):
    deposit_amount = 9.0
    alice_balance = plf_env.markets[dai_market_name].get_user(alice).unsupplied_amount
    execute_deposit(plf_env, alice, dai_market_name, deposit_amount)
    dai_market = plf_env.markets[dai_market_name]
    assert dai_market.user_atoken_balance(alice) == deposit_amount
    assert (
        plf_env.markets[dai_market_name].get_user(alice).unsupplied_amount
        == alice_balance - deposit_amount
    )
    assert dai_market.underlying_available == deposit_amount


def test_redeem(
    plf_env: PLFEnv,
    alice: str,
    dai_market_name: str,
    on_behalf_of: Optional[str] = None,
):
    on_behalf_of = alice
    alice_atoken_balance_previous = plf_env.markets[
        dai_market_name
    ].user_atoken_balance(alice)
    # print(plf_env.markets[dai_market_name].get_user(alice).atoken_balance)
    deposit_amount = 9
    redeem_amount = 9
    execute_deposit(
        plf_env, alice, dai_market_name, deposit_amount, on_behalf_of=on_behalf_of
    )
    # print(plf_env.markets[dai_market_name].get_user(alice).atoken_balance)
    redeem_action = Redeem(
        user=alice,
        on_behalf_of=on_behalf_of,
        market=dai_market_name,
        value=redeem_amount,
    )
    # print(plf_env.markets[dai_market_name].get_user(alice).atoken_balance)
    redeem_action.execute(plf_env)
    # print(plf_env.markets[dai_market_name].get_user(alice).atoken_balance)
    dai_market = plf_env.markets[dai_market_name]
    assert (
        dai_market.user_atoken_balance(alice)
        == alice_atoken_balance_previous + deposit_amount - redeem_amount
    )
    assert dai_market.underlying_available == deposit_amount - redeem_amount


@pytest.mark.parametrize("is_stable_borrow", [True, False])
def test_borrow(
    plf_env: PLFEnv,
    alice: str,
    bob: str,
    dai_market_name: str,
    bat_market_name: str,
    is_stable_borrow: bool,
):
    dai_deposit_amount = 9.0
    bat_deposit_amount = 5.0
    borrow_amount: float = 0.1
    borrow_var = "stable_borrow" if is_stable_borrow else "variable_borrow"

    # Alice deposits some DAI
    execute_deposit(plf_env, alice, dai_market_name, dai_deposit_amount)
    # Bob deposits some BAT
    execute_deposit(plf_env, bob, bat_market_name, bat_deposit_amount)

    dai_market = plf_env.markets[dai_market_name]

    deposit_price = plf_env.external_data.prices[dai_market_name]
    print(f"DAI price is {deposit_price} ========================================")
    collateral_factor = dai_market.collateral_factor
    borrow_price = plf_env.external_data.prices[bat_market_name]
    print(f"BAT price is {borrow_price} ========================================")

    # Alice borrow some BAT from the market
    execute_borrow(
        plf_env,
        user=alice,
        market=bat_market_name,
        amount=borrow_amount,
        is_stable_borrow=is_stable_borrow,
    )
    bat_market = plf_env.markets[bat_market_name]
    alice_account = bat_market.get_user(alice)

    assert getattr(alice_account, borrow_var) == borrow_amount
    assert bat_market.underlying_available == pytest.approx(
        bat_deposit_amount - borrow_amount
    )

    max_borrowable_amount = (
        deposit_price * dai_deposit_amount * collateral_factor / borrow_price
    )
    with pytest.raises(ValueError):
        execute_borrow(
            plf_env, alice, bat_market_name, max_borrowable_amount + 1, is_stable_borrow
        )  # exceeds max-borrowable, this should fail
    assert getattr(alice_account, borrow_var) == borrow_amount


@pytest.mark.parametrize("is_stable_borrow", [True, False])
def test_repay(
    plf_env: PLFEnv,
    alice: str,
    bob: str,
    dai_market_name: str,
    bat_market_name: str,
    is_stable_borrow: bool,
):
    dai_deposit_amount = 9
    bat_deposit_amount = 5
    borrow_amount = 2
    repay_amount = 1
    on_behalf_of = alice
    execute_deposit(
        plf_env, alice, dai_market_name, dai_deposit_amount, on_behalf_of=on_behalf_of
    )
    execute_deposit(plf_env, bob, bat_market_name, bat_deposit_amount)

    borrow_action = Borrow(
        user=alice,
        on_behalf_of=on_behalf_of,
        market=bat_market_name,
        value=borrow_amount,
        is_stable_borrow=is_stable_borrow,
    )
    borrow_action.execute(plf_env)
    market = plf_env.markets[bat_market_name]

    repay_action = Repay(
        user=alice,
        on_behalf_of=on_behalf_of,
        market=bat_market_name,
        value=repay_amount,
        is_stable_borrow=is_stable_borrow,
    )
    repay_action.execute(plf_env)

    expected_amount = borrow_amount - repay_amount
    if is_stable_borrow:
        assert market.get_user(alice).stable_borrow == expected_amount
    else:
        assert market.get_user(alice).variable_borrow == expected_amount


@pytest.mark.parametrize("is_stable_borrow", [True, False])
@pytest.mark.parametrize("receive_atoken", [True, False])
def test_liquidate(
    plf_env: PLFEnv,
    alice: str,
    bob: str,
    dai_market_name: str,
    bat_market_name: str,
    is_stable_borrow: bool,
    receive_atoken: bool,
):
    dai_deposit_amount = 10.0
    bat_deposit_amount = 5.0
    borrow_amount = 2.0
    repay_amount = 1.0
    bob_bat_balance = 100.0

    dai_market = plf_env.markets[dai_market_name]
    bat_market = plf_env.markets[bat_market_name]
    alice_bat_account = plf_env.markets[bat_market_name].get_user(alice)
    alice_dai_account = plf_env.markets[dai_market_name].get_user(alice)

    bob_bat_account = plf_env.markets[bat_market_name].get_user(bob)
    bob_dai_account = plf_env.markets[dai_market_name].get_user(bob)

    borrow_var = "stable_borrow" if is_stable_borrow else "variable_borrow"

    execute_deposit(plf_env, bob, bat_market_name, bat_deposit_amount)
    execute_deposit(plf_env, alice, dai_market_name, dai_deposit_amount)
    assert plf_env.markets[dai_market_name].get_user(alice).collateral_enabled

    execute_borrow(plf_env, alice, bat_market_name, borrow_amount, is_stable_borrow)
    assert getattr(alice_bat_account, borrow_var) == borrow_amount

    liquidation_action = Liquidate(
        bob,
        alice,
        bat_market_name,
        repay_amount,
        dai_market_name,
        receive_atoken=receive_atoken,
    )
    with pytest.raises(AssertionError):
        liquidation_action.execute(plf_env)

    assert getattr(alice_bat_account, borrow_var) == borrow_amount

    bat_price = plf_env.external_data.prices[
        bat_market_name
    ] = 6.0  # was previously 3.0
    dai_price = plf_env.external_data.prices[dai_market_name]

    liquidation_action.execute(plf_env)
    assert getattr(alice_bat_account, borrow_var) == borrow_amount - repay_amount
    assert (
        bob_bat_account.unsupplied_amount
        == bob_bat_balance - repay_amount - bat_deposit_amount
    )
    dai_delta = repay_amount * bat_price / dai_price / dai_market.liquidation_discount
    assert alice_dai_account.atoken_balance == 10 - dai_delta

    if receive_atoken:
        assert bob_dai_account.unsupplied_amount == 0
        assert bob_dai_account.atoken_balance == dai_delta
    else:
        assert bob_dai_account.unsupplied_amount == dai_delta
        assert bob_dai_account.atoken_balance == 0


@pytest.mark.parametrize("is_stable_borrow", [True, False])
def test_swap(
    plf_env: PLFEnv,
    alice: str,
    bob: str,
    dai_market_name: str,
    bat_market_name: str,
    is_stable_borrow: bool,
):
    alice_dai_borrow = 2
    plf_env.markets[dai_market_name].get_user(bob).unsupplied_amount = 10
    # Alice deposits some DAI
    execute_deposit(plf_env, alice, dai_market_name, 10)
    # Bob depoists some DAI
    execute_deposit(plf_env, bob, dai_market_name, 5)
    # Alice borrow some DAI
    execute_borrow(plf_env, alice, dai_market_name, alice_dai_borrow, is_stable_borrow)

    # # Alice borrow some BAT
    # TODO: check why we needed this
    # execute_borrow(plf_env, alice, bat_market_name, 3, is_stable_borrow)

    # Bob borrow some DAI
    execute_borrow(plf_env, bob, dai_market_name, 1, is_stable_borrow)

    dai_market = plf_env.markets[dai_market_name]
    if is_stable_borrow:
        init_borrow_var, final_borrow_var = "stable_borrow", "variable_borrow"
    else:
        init_borrow_var, final_borrow_var = "variable_borrow", "stable_borrow"
    assert getattr(dai_market.users[alice], init_borrow_var) == alice_dai_borrow
    assert getattr(dai_market.users[alice], final_borrow_var) == 0

    action = Swap(alice, dai_market_name, is_stable_borrow)
    action.execute(plf_env)
    assert getattr(dai_market.users[alice], init_borrow_var) == 0
    assert getattr(dai_market.users[alice], final_borrow_var) == alice_dai_borrow


def test_transfer(plf_env: PLFEnv, dai_market_name: str, alice: str, bob: str):
    dai_market = plf_env.markets[dai_market_name]
    execute_deposit(plf_env, alice, dai_market_name, 10)

    previous_alice_balance = dai_market.get_user(alice).atoken_balance
    previous_bob_balance = dai_market.get_user(bob).atoken_balance

    amount = 5
    action = Transfer(dai_market_name, alice, bob, amount)
    action.execute(plf_env)

    alice_balance = dai_market.get_user(alice).atoken_balance
    bob_balance = dai_market.get_user(bob).atoken_balance

    assert alice_balance == previous_alice_balance - amount
    assert bob_balance == previous_bob_balance + amount
